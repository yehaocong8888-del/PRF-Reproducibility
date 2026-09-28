import os, time, json, random
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model

MODEL_DIR = Path(os.environ["MODEL_DIR"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])
WORK_DIR  = Path(os.environ.get("WORK_DIR", str(Path.home() / "PRF_WORK_v1")))
OUT_F     = Path(os.environ["OUT_F"])

KEEP_JSON = Path(os.environ.get("KEEP_JSON", str(WORK_DIR / "keep_indices_trainA_v2.json")))

assert MODEL_DIR.exists(), MODEL_DIR
assert CACHE_DIR.exists(), CACHE_DIR
assert OUT_F.exists(), OUT_F
assert KEEP_JSON.exists(), KEEP_JSON

# ---- run config ----
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "100"))
LR         = float(os.environ.get("LR", "2e-4"))
DTYPE      = torch.float16
SEED       = int(os.environ.get("SEED", "42"))

# speed/memory toggles (default follow your successful 200: checkpointing ON)
USE_CKPT = os.environ.get("USE_CKPT", "1").strip() == "1"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")
assert device.type == "cuda"

# ✅ FIX: build run name safely (no % conflicts with strftime)
ts = time.strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"trainA_fast{MAX_STEPS:03d}_{ts}"
OUT_DIR  = OUT_F / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_JSONL = OUT_DIR / "train_log.jsonl"

print("[READ-ONLY] Cache is NOT modified. Training reads cache and writes only to OUT_DIR.")
print("[IN] MODEL_DIR:", MODEL_DIR)
print("[IN] CACHE_DIR:", CACHE_DIR)
print("[IN] KEEP_JSON:", KEEP_JSON)
print("[OUT] OUT_DIR  :", OUT_DIR)
print("[CFG] MAX_STEPS:", MAX_STEPS, "| LR:", LR, "| USE_CKPT:", USE_CKPT, "| SEED:", SEED)

keep_obj = json.loads(KEEP_JSON.read_text(encoding="utf-8"))
keep_indices = keep_obj["keep_indices"]
random.shuffle(keep_indices)

shards = sorted(CACHE_DIR.glob("train_shard_*.pt"))
assert shards, "No train_shard_*.pt found"
probe = torch.load(shards[0], map_location="cpu")
shard_size = len(probe)
num_shards = len(shards)
print(f"[CACHE] shards={num_shards} | shard_size={shard_size} | keep_total={len(keep_indices)}")

# bucket keep indices by shard
by_shard = defaultdict(list)
for idx in keep_indices:
    sid = idx // shard_size
    off = idx % shard_size
    if 0 <= sid < num_shards:
        by_shard[sid].append(off)

shard_ids = list(by_shard.keys())
random.shuffle(shard_ids)
print(f"[KEEP] shards_touched={len(shard_ids)}")

# model
print("[LOAD] tokenizer...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
pad_id = tok.pad_token_id or tok.eos_token_id

print("[LOAD] model...")
base = AutoModelForVision2Seq.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map="cuda",
)
base.train()
base.config.use_cache = False

if USE_CKPT:
    base.gradient_checkpointing_enable()

# freeze base
for p in base.parameters():
    p.requires_grad_(False)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)
model = get_peft_model(base, lora)
model.print_trainable_parameters()

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=True)

def make_batch(s):
    inp = s["input_ids"].to(torch.long)
    att = s["attention_mask"].to(torch.long)
    tgt = s["target_ids"].to(torch.long)

    # teacher forcing: concat prompt + target
    full_inp = torch.cat([inp, tgt], dim=0)
    full_att = torch.cat([att, torch.ones_like(tgt, dtype=torch.long)], dim=0)

    labels = torch.full_like(full_inp, -100)
    labels[len(inp):] = tgt

    return {
        "input_ids": full_inp.unsqueeze(0).to(device, non_blocking=True),
        "attention_mask": full_att.unsqueeze(0).to(device, non_blocking=True),
        "labels": labels.unsqueeze(0).to(device, non_blocking=True),
        "pixel_values": s["pixel_values"].unsqueeze(0).to(device, non_blocking=True),
        # IMPORTANT: keep (2,3). Do NOT stack to (B,2,3)
        "image_grid_thw": s["image_grid_thw"].to(device, non_blocking=True),
    }

model.train()
pbar = tqdm(total=MAX_STEPS, desc="TRAINA_FAST", dynamic_ncols=True, mininterval=0.5)
t0 = time.time()

step = 0
for sid in shard_ids:
    if step >= MAX_STEPS:
        break

    sp = shards[sid]
    data = torch.load(sp, map_location="cpu")  # one shard resident
    offs = by_shard[sid]
    random.shuffle(offs)

    for off in offs:
        if step >= MAX_STEPS:
            break

        batch = make_batch(data[off])

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=DTYPE):
            out = model(**batch)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        step += 1
        dt = time.time() - t0
        sps = step / max(1e-9, dt)

        if (step % 10) == 0 or step == 1:
            msg = {"step": step, "loss": float(loss.detach().float().cpu()), "sps": sps, "shard": sid}
            with LOG_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        pbar.set_postfix(loss=f"{float(loss.detach().float().cpu()):.4f}", sps=f"{sps:.2f}", shard=str(sid))
        pbar.update(1)

pbar.close()

print("[SAVE] saving LoRA adapter to:", OUT_DIR)
model.save_pretrained(OUT_DIR)

print("[DONE] trainA fast run complete")
print("[OUT]", OUT_DIR)
print("[LOG]", LOG_JSONL)
