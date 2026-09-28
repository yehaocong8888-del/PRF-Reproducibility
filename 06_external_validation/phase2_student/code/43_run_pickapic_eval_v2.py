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
OUT_DIR = Path(os.environ["OUT_DIR"])

OUT_PRED = OUT_DIR / "pickapic_pred_v2.jsonl"
OUT_DONE = OUT_DIR / "pickapic_pred_v2.done"
OUT_ERR  = OUT_DIR / "pickapic_pred_v2_errors.jsonl"

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
    return f"""You are an expert evaluator of text-to-image preference.

Caption:
{caption}

Task:
Given Image A and Image B, choose which image better matches the caption.

Rules:
- Consider semantic correctness first
- Then consider visual quality and coherence
- Answer with only one letter: A or B
- Do not explain
""".strip()

def normalize_pred(text: str) -> str:
    t = text.strip().upper()
    if t == "A":
        return "A"
    if t == "B":
        return "B"

    # 容错一点，但不要太宽
    if t.startswith("A") and not t.startswith("AB"):
        return "A"
    if t.startswith("B") and not t.startswith("BA"):
        return "B"

    if "A" in t and "B" not in t:
        return "A"
    if "B" in t and "A" not in t:
        return "B"

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

    pred_dist = Counter()
    valid = 0
    correct = 0

    with open(OUT_PRED, "a", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="pickapic eval v2", ncols=100):
            pid = row["pair_id"]
            if pid in done:
                continue

            try:
                imgA = Image.open(row["img_A"]).convert("RGB")
                imgB = Image.open(row["img_B"]).convert("RGB")
                prompt = build_prompt(row["caption"])

                # 关键：必须用 chat template 显式放入两个 image token
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
                    images=[imgA, imgB],
                    padding=True,
                    return_tensors="pt"
                )

                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False
                    )

                # 只解码新生成部分，不解码整段 prompt
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                raw_text = processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True
                )[0].strip()

                pred = normalize_pred(raw_text)
                ok = int(pred == row["gt"])

                pred_dist[pred] += 1
                if pred != "UNK":
                    valid += 1
                correct += ok

                out = dict(row)
                out["raw_text"] = raw_text
                out["pred"] = pred
                out["correct"] = ok

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                append_done(OUT_DONE, pid)

            except Exception as e:
                append_err(OUT_ERR, {
                    "pair_id": pid,
                    "img_A": row.get("img_A"),
                    "img_B": row.get("img_B"),
                    "caption": row.get("caption"),
                    "error": str(e),
                })

    print("[DONE] pred ->", OUT_PRED)
    print("[DONE] err  ->", OUT_ERR)
    print("[INFO] pred_dist =", dict(pred_dist))
    print("[INFO] valid =", valid)
    print("[INFO] correct =", correct)

    if valid > 0:
        print("[INFO] acc(valid) =", correct / valid)
    else:
        print("[WARN] no valid predictions")

if __name__ == "__main__":
    main()
