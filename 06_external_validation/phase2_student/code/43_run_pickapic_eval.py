# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

# ===== 路径 =====
BASE_MODEL = os.environ["BASE_MODEL"]
LORA_MODEL = os.environ["LORA_MODEL"]
PAIRS_JSONL = os.environ["PAIRS_TEST"]
OUT_PATH = os.environ["OUT_DIR"] + "/pickapic_pred.jsonl"

# ===== 加载模型 =====
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

# ===== Prompt（关键）=====
def build_prompt(caption):
    return f"""You are an expert evaluator of image quality.

A text prompt is given, along with two images.

Caption:
{caption}

Two images:
Image A
Image B

Task:
Which image better matches the caption?

Rules:
- Consider semantic correctness
- Consider visual quality
- Answer ONLY one letter: A or B
- Do not explain
"""

# ===== 推理 =====
def run_one(row):
    imgA = Image.open(row["img_A"]).convert("RGB")
    imgB = Image.open(row["img_B"]).convert("RGB")

    prompt = build_prompt(row["caption"])

    inputs = processor(
        text=prompt,
        images=[imgA, imgB],
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    if "A" in text and "B" not in text:
        pred = "A"
    elif "B" in text and "A" not in text:
        pred = "B"
    else:
        pred = "UNK"

    return pred, text

# ===== 主流程 =====
def main():
    rows = []
    with open(PAIRS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print("[INFO] total =", len(rows))

    correct = 0
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for r in tqdm(rows):
            pred, raw = run_one(r)

            ok = (pred == r["gt"])
            if ok:
                correct += 1
            total += 1

            r["pred"] = pred
            r["raw_text"] = raw
            r["correct"] = int(ok)

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / total
    print(f"[DONE] acc = {acc:.4f} ({correct}/{total})")
    print("[OUT]", OUT_PATH)

if __name__ == "__main__":
    main()
