# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

BASE_MODEL = os.environ["BASE_MODEL"]
LORA_MODEL = os.environ["LORA_MODEL"]
PAIRS_JSONL = os.environ["PAIRS_TEST"]
OUT_DIR = Path(os.environ["PRF_EVAL_DIR"])

OUT_PRED = OUT_DIR / "pickapic_prf_pred.jsonl"
OUT_DONE = OUT_DIR / "pickapic_prf_pred.done"
OUT_ERR  = OUT_DIR / "pickapic_prf_pred_errors.jsonl"

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def read_done(path):
    if not path.exists():
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

def append_done(path, pid):
    with open(path, "a", encoding="utf-8") as f:
        f.write(pid + "\n")

def append_err(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_prompt(caption: str) -> str:
    return f"""You are an image-pair judge for training data selection.

A text prompt is given:
{caption}

Two images are given:
- Query image
- Candidate image

Task:
Decide whether the candidate image should be ACCEPTED or REJECTED relative to the query image, based on which image better matches the text prompt.

Decision rule:
- ACCEPT: candidate is better than query
- REJECT: candidate is worse than query

Output rule:
- Output only one decision token
- Prefer the PRF decision style
""".strip()

def parse_pred(raw_text: str) -> str:
    t = raw_text.strip().upper()

    # 优先解析 PRF-native
    if "DEC=A" in t:
        return "ACCEPT"
    if "DEC=R" in t:
        return "REJECT"
    if "DEC=U" in t:
        return "AMBIGUOUS"

    # 容错
    if t == "ACCEPT":
        return "ACCEPT"
    if t == "REJECT":
        return "REJECT"
    if t == "A":
        return "ACCEPT"
    if t == "R":
        return "REJECT"
    if t == "U":
        return "AMBIGUOUS"

    return "UNK"

def main():
    print("[INFO] loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print("[INFO] loading base model...")
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("[INFO] loading LoRA...")
    model = PeftModel.from_pretrained(model, LORA_MODEL)
    model.eval()

    rows = load_rows(PAIRS_JSONL)
    done = read_done(OUT_DONE)

    print("[INFO] total rows =", len(rows))
    print("[INFO] resume done =", len(done))

    gt_dist = Counter()
    pred_dist = Counter()
    valid = 0
    correct = 0
    total = 0

    with open(OUT_PRED, "a", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="pickapic prf eval", ncols=100):
            pid = row["pair_id"]
            if pid in done:
                continue

            total += 1
            gt_dist[row["gt"]] += 1

            try:
                imgQ = Image.open(row["query_path"]).convert("RGB")
                imgC = Image.open(row["candidate_path"]).convert("RGB")
                prompt = build_prompt(row["caption"])

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=[text],
                    images=[imgQ, imgC],
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=12,
                        do_sample=False
                    )

                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                raw_text = processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True
                )[0].strip()

                pred = parse_pred(raw_text)
                pred_dist[pred] += 1

                ok = int(pred == row["gt"])
                if pred in ("ACCEPT", "REJECT"):
                    valid += 1
                    correct += ok

                out = dict(row)
                out["raw_text"] = raw_text
                out["pred"] = pred
                out["correct"] = ok if pred in ("ACCEPT", "REJECT") else None

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                append_done(OUT_DONE, pid)

            except Exception as e:
                append_err(OUT_ERR, {
                    "pair_id": pid,
                    "query_path": row.get("query_path"),
                    "candidate_path": row.get("candidate_path"),
                    "caption": row.get("caption"),
                    "error": str(e),
                })

    print("[DONE] pred ->", OUT_PRED)
    print("[DONE] err  ->", OUT_ERR)
    print("[INFO] gt_dist   =", dict(gt_dist))
    print("[INFO] pred_dist =", dict(pred_dist))
    print("[INFO] total =", total)
    print("[INFO] valid =", valid)
    print("[INFO] correct =", correct)
    if valid > 0:
        print("[INFO] acc(valid) =", correct / valid)
    else:
        print("[WARN] no valid predictions")

if __name__ == "__main__":
    main()
