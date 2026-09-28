# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

BASE_MODEL = os.environ["BASE_MODEL"]
PAIRS_JSONL = os.environ["PAIRS_TEST"]
OUT_DIR = Path(os.environ["BASE_ABL_DIR"])

OUT_PRED = OUT_DIR / "base8b_prompt_ablation_preds.jsonl"
OUT_SUMMARY = OUT_DIR / "base8b_prompt_ablation_summary.json"

PROMPTS = {
    "V1_direct_binary": """You are judging whether the candidate image should be accepted relative to the query image.

Caption:
{caption}

Two images are given:
- Query image
- Candidate image

Decision rule:
- ACCEPT: the candidate image matches the caption better than the query image
- REJECT: the candidate image matches the caption worse than the query image

Rules:
- You must choose one of ACCEPT or REJECT
- Do not output AMBIGUOUS
- Output only the final decision
""".strip(),

    "V2_prf_style_noU": """You are an image-pair judge.

Caption:
{caption}

Two images are given:
- Query image
- Candidate image

Decision rule:
- ACCEPT: candidate is better than query for the caption
- REJECT: candidate is worse than query for the caption

Output rule:
- Use PRF decision style
- Output only one decision
- Allowed decisions: DEC=A or DEC=R
- Do not output DEC=U
""".strip(),

    "V3_winner_loser": """Caption:
{caption}

Compare the query image and the candidate image.

If the candidate is the better image for the caption, output ACCEPT.
If the candidate is the worse image for the caption, output REJECT.

Output exactly one word:
ACCEPT
or
REJECT
""".strip(),
}

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def parse_pred(raw_text: str) -> str:
    t = raw_text.strip().upper()

    if "DEC=A" in t:
        return "ACCEPT"
    if "DEC=R" in t:
        return "REJECT"
    if "DEC=U" in t:
        return "AMBIGUOUS"

    if t == "ACCEPT":
        return "ACCEPT"
    if t == "REJECT":
        return "REJECT"
    if t == "AMBIGUOUS":
        return "AMBIGUOUS"

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

    print("[INFO] loading base model only...")
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    rows = load_rows(PAIRS_JSONL)
    print("[INFO] total rows =", len(rows))

    all_preds = []
    summary = {}

    for prompt_name, prompt_tpl in PROMPTS.items():
        print(f"\n[RUN] {prompt_name}")
        gt_dist = Counter()
        pred_dist = Counter()
        valid = 0
        correct = 0
        total = 0

        for row in tqdm(rows, desc=f"base8b_{prompt_name}", ncols=100):
            total += 1
            gt_dist[row["gt"]] += 1

            imgQ = Image.open(row["query_path"]).convert("RGB")
            imgC = Image.open(row["candidate_path"]).convert("RGB")

            prompt = prompt_tpl.format(caption=row["caption"])

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

            all_preds.append({
                "model": "base8b",
                "prompt_name": prompt_name,
                "pair_id": row["pair_id"],
                "gt": row["gt"],
                "pred": pred,
                "correct": ok if pred in ("ACCEPT", "REJECT") else None,
                "raw_text": raw_text,
                "caption": row["caption"],
                "query_path": row["query_path"],
                "candidate_path": row["candidate_path"],
            })

        acc = (correct / valid) if valid > 0 else None
        summary[prompt_name] = {
            "gt_dist": dict(gt_dist),
            "pred_dist": dict(pred_dist),
            "total": total,
            "valid": valid,
            "correct": correct,
            "acc_valid": acc,
        }

        print("[INFO]", prompt_name, "gt_dist   =", dict(gt_dist))
        print("[INFO]", prompt_name, "pred_dist =", dict(pred_dist))
        print("[INFO]", prompt_name, "valid =", valid)
        print("[INFO]", prompt_name, "correct =", correct)
        print("[INFO]", prompt_name, "acc(valid) =", acc)

    with open(OUT_PRED, "w", encoding="utf-8") as f:
        for r in all_preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE] preds ->", OUT_PRED)
    print("[DONE] summary ->", OUT_SUMMARY)

    print("\n[FINAL SUMMARY]")
    for k, v in summary.items():
        print(k, "acc(valid) =", v["acc_valid"], "| valid =", v["valid"], "| pred_dist =", v["pred_dist"])

if __name__ == "__main__":
    main()
