import os, time, json, random
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForVision2Seq
from peft import PeftModel

MODEL_DIR = Path(os.environ["MODEL_DIR"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])
WORK_DIR  = Path(os.environ.get("WORK_DIR", str(Path.home() / "PRF_WORK_v1")))
OUT_F     = Path(os.environ["OUT_F"])
KEEP_JSON = Path(os.environ.get("KEEP_JSON", str(WORK_DIR / "keep_indices_trainA_v2.json")))

OLD_RUN_DIR = Path(os.environ.get(
    "OLD_RUN_DIR",
    "/mnt/f/PRF/loraUbuntu/GCSJ/prf_vl_student_v1/trainA_full_keep20260227_234225"
))
RESUME_CKPT = OLD_RUN_DIR / "ckpt_step020000"

assert MODEL_DIR.exists(), MODEL_DIR
assert CACHE_DIR.exists(), CACHE_DIR
assert OUT_F.exists(), OUT_F
assert KEEP_JSON.exists(), KEEP_JSON
assert OLD_RUN_DIR.exists(), OLD_RUN_DIR
assert RESUME_CKPT.exists(), RESUME_CKPT

SEED        = int(os.environ.get("SEED", "42"))
LR          = float(os.environ.get("LR", "1e-4"))
USE_CKPT    = os.environ.get("USE_CKPT", "1").strip() == "1"
SAVE_EVERY  = int(os.environ.get("SAVE_EVERY", "500"))
LOG_EVERY   = int(os.environ.get("LOG_EVERY", "20"))
GRAD_CLIP   = float(os.environ.get("GRAD_CLIP", "1.0"))
START_STEP  = int(os.environ.get("START_STEP", "20000"))
TARGET_STEP = int(os.environ.get("TARGET_STEP", "26607"))
MAX_EXTRA_SKIPS = int(os.environ.get("MAX_EXTRA_SKIPS", "200"))

assert TARGET_STEP > START_STEP, (START_STEP, TARGET_STEP)
assert torch.cuda.is_available(), "CUDA not available"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")
AMP_DTYPE = torch.bfloat16   # ✅ 关键：改成 bf16，更稳

ts = time.strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"resume_from020000_bf16_{ts}"
OUT_DIR  = OUT_F / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_JSONL = OUT_DIR / "train_log.jsonl"

print("[READ-ONLY] cache/keep/old ckpt are NOT modified. Writes only to new OUT_DIR.")
print("[IN]  MODEL_DIR   :", MODEL_DIR)
print("[IN]  CACHE_DIR   :", CACHE_DIR)
print("[IN]  KEEP_JSON   :", KEEP_JSON)
print("[IN]  OLD_RUN_DIR :", OLD_RUN_DIR)
print("[IN]  RESUME_CKPT :", RESUME_CKPT)
print("[OUT] OUT_DIR     :", OUT_DIR)
print("[CFG] LR:", LR, "| USE_CKPT:", USE_CKPT, "| SAVE_EVERY:", SAVE_EVERY,
      "| LOG_EVERY:", LOG_EVERY, "| GRAD_CLIP:", GRAD_CLIP,
      "| START_STEP:", START_STEP, "| TARGET_STEP:", TARGET_STEP,
      "| MAX_EXTRA_SKIPS:", MAX_EXTRA_SKIPS, "| AMP_DTYPE: bf16")

keep_obj = json.loads(KEEP_JSON.read_text(encoding="utf-8"))
keep_indices = keep_obj["keep_indices"][:]

shards = sorted(CACHE_DIR.glob("train_shard_*.pt"))
assert shards, "No train_shard_*.pt found"
probe = torch.load(shards[0], map_location="cpu")
shard_size = len(probe)
num_shards = len(shards)
print(f"[CACHE] shards={num_shards} | shard_size={shard_size} | keep_total={len(keep_indices)}")

# ---- 重建原始调度 ----
random.seed(SEED)
random.shuffle(keep_indices)

by_shard = defaultdict(list)
for idx in keep_indices:
    sid = idx // shard_size
    off = idx % shard_size
    if 0 <= sid < num_shards:
        by_shard[sid].append(off)

shard_ids = list(by_shard.keys())
random.shuffle(shard_ids)

schedule = []
for sid in shard_ids:
    offs = by_shard[sid][:]
    random.shuffle(offs)
    for off in offs:
        schedule.append((sid, off))

assert len(schedule) >= TARGET_STEP, (len(schedule), TARGET_STEP)
resume_schedule = schedule[START_STEP:TARGET_STEP]
print(f"[SCHEDULE] full={len(schedule)} | resume_len={len(resume_schedule)}")

print("[LOAD] tokenizer...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

print("[LOAD] base model...")
base = AutoModelForVision2Seq.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    torch_dtype=torch.float16,   # 权重仍沿用原加载方式
    device_map="cuda",
)
base.train()
base.config.use_cache = False

if USE_CKPT:
    base.gradient_checkpointing_enable()

print("[LOAD] LoRA adapter from ckpt_step020000 ...")
model = PeftModel.from_pretrained(
    base,
    str(RESUME_CKPT),
    is_trainable=True,
)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

def make_batch(s):
    inp = s["input_ids"].to(torch.long)
    att = s["attention_mask"].to(torch.long)
    tgt = s["target_ids"].to(torch.long)

    full_inp = torch.cat([inp, tgt], dim=0)
    full_att = torch.cat([att, torch.ones_like(tgt, dtype=torch.long)], dim=0)

    labels = torch.full_like(full_inp, -100)
    labels[len(inp):] = tgt

    return {
        "input_ids": full_inp.unsqueeze(0).to(device, non_blocking=True),
        "attention_mask": full_att.unsqueeze(0).to(device, non_blocking=True),
        "labels": labels.unsqueeze(0).to(device, non_blocking=True),
        "pixel_values": s["pixel_values"].unsqueeze(0).to(device, non_blocking=True),
        "image_grid_thw": s["image_grid_thw"].to(device, non_blocking=True),
    }

def save_ckpt(tag: str):
    ckpt_dir = OUT_DIR / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    return ckpt_dir

(OUT_DIR / "resume_meta.json").write_text(json.dumps({
    "resume_from": str(RESUME_CKPT),
    "old_run_dir": str(OLD_RUN_DIR),
    "keep_json": str(KEEP_JSON),
    "seed": SEED,
    "lr": LR,
    "use_ckpt": USE_CKPT,
    "save_every": SAVE_EVERY,
    "log_every": LOG_EVERY,
    "grad_clip": GRAD_CLIP,
    "start_step": START_STEP,
    "target_step": TARGET_STEP,
    "resume_schedule_len": len(resume_schedule),
    "amp_dtype": "bfloat16",
}, ensure_ascii=False, indent=2), encoding="utf-8")

target_updates = TARGET_STEP - START_STEP
pbar = tqdm(total=target_updates, desc="RESUME_FULL", dynamic_ncols=True, mininterval=0.5)

t0 = time.time()
global_step = START_STEP
local_update = 0
skip_nonfinite = 0
loaded_sid = None
loaded_data = None

for sid, off in resume_schedule:
    if global_step >= TARGET_STEP:
        break

    if loaded_sid != sid:
        loaded_data = torch.load(shards[sid], map_location="cpu")
        loaded_sid = sid

    batch = make_batch(loaded_data[off])
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        out = model(**batch)
        loss = out.loss

    if not torch.isfinite(loss):
        skip_nonfinite += 1
        print(f"\n[WARN] skip non-finite loss at planned_step={global_step+1}, shard={sid}, off={off}, skip_count={skip_nonfinite}")
        opt.zero_grad(set_to_none=True)
        if skip_nonfinite > MAX_EXTRA_SKIPS:
            print("[STOP] too many non-finite batches, aborting.")
            break
        continue

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        skip_nonfinite += 1
        print(f"\n[WARN] skip non-finite grad_norm at planned_step={global_step+1}, shard={sid}, off={off}, skip_count={skip_nonfinite}")
        opt.zero_grad(set_to_none=True)
        if skip_nonfinite > MAX_EXTRA_SKIPS:
            print("[STOP] too many non-finite grad batches, aborting.")
            break
        continue

    opt.step()

    global_step += 1
    local_update += 1

    dt = time.time() - t0
    sps = local_update / max(1e-9, dt)

    if (global_step % LOG_EVERY) == 0 or global_step == START_STEP + 1:
        msg = {
            "global_step": global_step,
            "local_update": local_update,
            "loss": float(loss.detach().float().cpu()),
            "sps": sps,
            "shard": sid,
            "grad_norm": float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "skip_nonfinite": skip_nonfinite,
        }
        with LOG_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    pbar.set_postfix(
        loss=f"{float(loss.detach().float().cpu()):.4f}",
        gnorm=f"{float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm):.2f}",
        sps=f"{sps:.2f}",
        shard=str(sid),
        skips=str(skip_nonfinite),
    )
    pbar.update(1)

    if SAVE_EVERY > 0 and (global_step % SAVE_EVERY) == 0:
        tag = f"ckpt_step{global_step:06d}"
        ckpt_dir = save_ckpt(tag)
        print(f"\n[SAVE] {tag} -> {ckpt_dir}")

pbar.close()

final_dir = save_ckpt("final")
print("[SAVE] final ->", final_dir)
print("[DONE] resume training complete")
print("[OUT]", OUT_DIR)
print("[LOG]", LOG_JSONL)
print("[SUMMARY] finished_global_step:", global_step, "| target_step:", TARGET_STEP, "| skip_nonfinite:", skip_nonfinite)
