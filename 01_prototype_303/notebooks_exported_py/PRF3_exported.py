# AUTO-EXPORTED FROM ORIGINAL JUPYTER NOTEBOOK
# This file is DERIVED and was not executed during export.
# Source notebook: PRF3.ipynb

# %% [cell 1]
# ===== Cell: Backfill cap stats for v1-v4 and merge into 3-criteria table (NO only exists in v4) =====
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -------- fixed dirs (与你截图一致) --------
CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
OUTS = Path(r"<PROTOTYPE303_ROOT>\outputs")
OUTS.mkdir(parents=True, exist_ok=True)

# -------- inputs --------
IN_TABLE = OUTS / "table_3criteria_v1_v2_v3_v3_1_v4.csv"
assert IN_TABLE.exists() and IN_TABLE.stat().st_size > 0, f"Missing/empty: {IN_TABLE}"

VERS = ["v1", "v2", "v3", "v3_1", "v4"]
CAP = 0.2  # 你们当前论文口径用的 cap

def load_cover_fracs(poly_jsonl: Path):
    """从 b2_polygon_allpairs_v*.jsonl 里读取 polygon_stats.cover_frac"""
    assert poly_jsonl.exists() and poly_jsonl.stat().st_size > 0, f"Missing/empty: {poly_jsonl}"

    vals = []
    with poly_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # 你 notebook 里统计 cover_frac 的入口就是 polygon_stats.cover_frac
            cf = (r.get("polygon_stats") or {}).get("cover_frac", None)
            if cf is None:
                continue
            try:
                vals.append(float(cf))
            except:
                continue
    return np.asarray(vals, dtype=float)

def summarize_cap(cover_fracs: np.ndarray, cap: float):
    """输出 raw / capped 的统计 + share_hit_cap（不引入 share_NO，v4 的 NO 另做审计）"""
    assert cover_fracs.ndim == 1
    if len(cover_fracs) == 0:
        return {
            "cover_frac_raw": {"n": 0},
            "cover_frac_capped": {"cap_value": cap, "n": 0},
        }

    raw = cover_fracs
    capped = np.minimum(raw, cap)
    hit = (raw > cap)

    raw_stats = {
        "n": int(len(raw)),
        "mean": float(np.mean(raw)),
        "p50": float(np.percentile(raw, 50)),
        "p90": float(np.percentile(raw, 90)),
        "p95": float(np.percentile(raw, 95)),
        "max": float(np.max(raw)),
    }
    cap_stats = {
        "cap_value": float(cap),
        "n": int(len(capped)),
        "mean_cap": float(np.mean(capped)),
        "p50_cap": float(np.percentile(capped, 50)),
        "p90_cap": float(np.percentile(capped, 90)),
        "p95_cap": float(np.percentile(capped, 95)),
        "max_cap": float(np.max(capped)),
        "share_hit_cap": float(np.mean(hit)),
    }
    return {"cover_frac_raw": raw_stats, "cover_frac_capped": cap_stats}

# -------- 1) per-version: compute + save summary json --------
summary_paths = {}
for v in VERS:
    poly_file = CKPT / f"b2_polygon_allpairs_{v}.jsonl"
    cover = load_cover_fracs(poly_file)
    s = summarize_cap(cover, CAP)

    out_json = OUTS / f"summary_{v}_coverfrac_capped.json"
    out_json.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_paths[v] = out_json

    print(f"[DONE] {v}: n={s['cover_frac_raw'].get('n',0)} | saved -> {out_json}")

print("\n[INFO] summary json paths:")
for v in VERS:
    print(" ", v, "=>", summary_paths[v])

# -------- 2) merge into table_3criteria (one row per version) --------
df = pd.read_csv(IN_TABLE, encoding="utf-8")
assert "version" in df.columns, "Input table must contain column: version"

# 统一列名：与你之前 v4 merge 代码一致（按 cap 值动态命名）
col_p90_cap = f"Dispute proxy: cover_frac_p90_cap{float(CAP)}"
col_share_hit = f"Dispute proxy: share_hit_cap{float(CAP)}"

if col_p90_cap not in df.columns:
    df[col_p90_cap] = ""
if col_share_hit not in df.columns:
    df[col_share_hit] = ""

for v in VERS:
    s = json.loads(summary_paths[v].read_text(encoding="utf-8"))
    p90_cap = float(s["cover_frac_capped"].get("p90_cap", np.nan))
    share_hit_cap = float(s["cover_frac_capped"].get("share_hit_cap", np.nan))

    mask = df["version"].astype(str).str.lower().eq(v.lower())
    assert mask.any(), f"No row found for version={v} in table (check df['version'])."

    df.loc[mask, col_p90_cap] = p90_cap
    df.loc[mask, col_share_hit] = share_hit_cap

# numeric cleanup
df[col_p90_cap] = pd.to_numeric(df[col_p90_cap], errors="coerce")
df[col_share_hit] = pd.to_numeric(df[col_share_hit], errors="coerce")

OUT_TABLE = OUTS / "table_3criteria_v1_v2_v3_v3_1_v4_withcap.csv"
df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

print("\n[DONE] merged v1-v4 capped columns into table")
print("[IN ]", IN_TABLE)
print("[OUT]", OUT_TABLE)
print("\n[PREVIEW]")
print(df)

# %% [cell 2]
from pathlib import Path
import json

p = Path(r"<PROTOTYPE303_ROOT>\checkpoints\b2_polygon_allpairs_v4.jsonl")

with p.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        print(r.keys())
        print(json.dumps(r, indent=2, ensure_ascii=False)[:1500])
        break

# %% [cell 3]
from pathlib import Path
import json

p = Path(r"<PROTOTYPE303_ROOT>\checkpoints\b2_polygon_allpairs_v4.jsonl")

with p.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        print(r.keys())
        print(json.dumps(r, indent=2, ensure_ascii=False)[:1500])
        break

# %% [cell 4]
# ===== OVERWRITE: recompute cover_frac from polygons.points (normalized), then cap stats, then merge =====
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------ fixed paths (与你截图一致) ------------------
CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
OUTS = Path(r"<PROTOTYPE303_ROOT>\outputs")
OUTS.mkdir(parents=True, exist_ok=True)

IN_TABLE = OUTS / "table_3criteria_v1_v2_v3_v3_1_v4.csv"
assert IN_TABLE.exists() and IN_TABLE.stat().st_size > 0, f"Missing/empty: {IN_TABLE}"

VERS = ["v1", "v2", "v3", "v3_1", "v4"]
CAP_VALUE = 0.2  # 与你们 v4 capped summary 口径一致

# ------------------ geometry: polygon area on unit square ------------------
def polygon_area(points):
    """
    points: list of [x,y] in normalized coords (0~1)
    Shoelace formula.
    """
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2 - x2 * y1)
    return abs(area) * 0.5

def compute_cover_frac_from_row(r):
    """
    Your schema (v4 example):
    r['polygons'] = [{label, points:[[x,y],...], reason_zh}, ...]
    points are normalized -> area is already fraction.
    cover_frac = sum polygon areas, clipped to [0,1]
    """
    polys = r.get("polygons", [])
    if not polys:
        return 0.0

    total = 0.0
    for p in polys:
        pts = p.get("points", [])
        # robust parse to float
        try:
            pts_f = [[float(a), float(b)] for a, b in pts]
        except Exception:
            pts_f = []
        total += polygon_area(pts_f)

    # clip to [0,1]
    if total < 0:
        total = 0.0
    if total > 1:
        total = 1.0
    return float(total)

# ------------------ io helpers ------------------
def read_jsonl(path: Path):
    assert path.exists() and path.stat().st_size > 0, f"Missing/empty: {path}"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def qstats(x: np.ndarray):
    if x.size == 0:
        return dict(n=0, mean=np.nan, p50=np.nan, p90=np.nan, p95=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        p50=float(np.percentile(x, 50)),
        p90=float(np.percentile(x, 90)),
        p95=float(np.percentile(x, 95)),
        max=float(np.max(x)),
    )

# ------------------ per-version compute + OVERWRITE summaries ------------------
summary_paths = {}
for v in VERS:
    poly_path = CKPT / f"b2_polygon_allpairs_{v}.jsonl"
    rows = read_jsonl(poly_path)

    cover_raw = np.asarray([compute_cover_frac_from_row(r) for r in rows], dtype=float)
    cover_cap = np.minimum(cover_raw, CAP_VALUE)
    hit_cap = (cover_raw > CAP_VALUE)

    out = {
        "cover_frac_raw": qstats(cover_raw),
        "cover_frac_capped": {
            "cap_value": float(CAP_VALUE),
            "mean_cap": float(np.mean(cover_cap)) if cover_cap.size else np.nan,
            "p50_cap": float(np.percentile(cover_cap, 50)) if cover_cap.size else np.nan,
            "p90_cap": float(np.percentile(cover_cap, 90)) if cover_cap.size else np.nan,
            "p95_cap": float(np.percentile(cover_cap, 95)) if cover_cap.size else np.nan,
            "max_cap": float(np.max(cover_cap)) if cover_cap.size else np.nan,
            "share_hit_cap": float(np.mean(hit_cap)) if hit_cap.size else np.nan,
            "n": int(cover_cap.size),
        },
    }

    # v4 only: share_NO (noise_decision field exists only in v4 by your design)
    if v == "v4":
        decs = []
        for r in rows:
            d = r.get("noise_decision", None)
            if isinstance(d, str):
                decs.append(d.strip().upper())
        if len(decs) > 0:
            out["aux_v4"] = {
                "share_NO": float(np.mean([d == "NO" for d in decs])),
                "share_YES": float(np.mean([d == "YES" for d in decs])),
                "n_with_decision": int(len(decs)),
            }
        else:
            out["aux_v4"] = {"share_NO": np.nan, "share_YES": np.nan, "n_with_decision": 0}

    # OVERWRITE to the same filename to avoid confusion
    out_json = OUTS / f"summary_{v}_coverfrac_capped.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_paths[v] = out_json

    print(f"[DONE] {v}: n={out['cover_frac_raw']['n']} | overwrite -> {out_json}")

# ------------------ merge into 3-criteria table (OVERWRITE output file) ------------------
df = pd.read_csv(IN_TABLE, encoding="utf-8")
assert "version" in df.columns, "Input table must contain column: version"

# Columns: keep your existing naming convention
col_p90_cap = f"Dispute proxy: cover_frac_p90_cap{CAP_VALUE}"
col_share_hit = f"Dispute proxy: share_hit_cap{CAP_VALUE}"
col_share_no = "Aux: share_NO (v4 only)"

for c in [col_p90_cap, col_share_hit, col_share_no]:
    if c not in df.columns:
        df[c] = np.nan

for v in VERS:
    s = json.loads(summary_paths[v].read_text(encoding="utf-8"))
    p90_cap = s["cover_frac_capped"].get("p90_cap", np.nan)
    share_hit_cap = s["cover_frac_capped"].get("share_hit_cap", np.nan)

    mask = df["version"].astype(str).str.lower().eq(v.lower())
    assert mask.any(), f"No row found for version={v} in table"

    df.loc[mask, col_p90_cap] = p90_cap
    df.loc[mask, col_share_hit] = share_hit_cap

    if v == "v4":
        share_no = (s.get("aux_v4") or {}).get("share_NO", np.nan)
        df.loc[mask, col_share_no] = share_no
    else:
        df.loc[mask, col_share_no] = np.nan  # must be N/A for v1-v3.1

# force numeric types where applicable
df[col_p90_cap] = pd.to_numeric(df[col_p90_cap], errors="coerce")
df[col_share_hit] = pd.to_numeric(df[col_share_hit], errors="coerce")
df[col_share_no] = pd.to_numeric(df[col_share_no], errors="coerce")

OUT_TABLE = OUTS / "table_3criteria_v1_v2_v3_v3_1_v4_withcap.csv"
df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

print("\n[DONE] OVERWRITE merge completed")
print("[IN ]", IN_TABLE)
print("[OUT]", OUT_TABLE)
print("\n[PREVIEW]\n", df)

# %% [cell 5]
# =========================================================
# PRF 303 / Step A (Transformers CLIP, local DIRECTORY)
# Compute A-CLIP on fixed candidate pool and dump results
# =========================================================

import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# 路径（冻结）
# -----------------------------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
OUT_FILE   = CKPT / "a_clip_on_pool_303w.jsonl"
ERR_FILE   = CKPT / "a_clip_on_pool_303w_errors.jsonl"

assert PAIRS_FILE.exists(), f"Missing: {PAIRS_FILE}"

# -----------------------------
# 关键：直接指向你截图的本地目录（这里就是你给过的路径）
# -----------------------------
LOCAL_CLIP_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert LOCAL_CLIP_DIR.exists(), f"Missing local CLIP dir: {LOCAL_CLIP_DIR}"
assert (LOCAL_CLIP_DIR / "config.json").exists(), "Missing config.json"
assert (LOCAL_CLIP_DIR / "model.safetensors").exists(), "Missing model.safetensors"
assert (LOCAL_CLIP_DIR / "preprocessor_config.json").exists(), "Missing preprocessor_config.json"

# -----------------------------
# 覆盖输出（你要求覆盖，避免混淆）
# -----------------------------
OVERWRITE = True
if OVERWRITE:
    if OUT_FILE.exists(): OUT_FILE.unlink()
    if ERR_FILE.exists(): ERR_FILE.unlink()
    print("[INFO] overwrite=True -> cleared old outputs")

# -----------------------------
# 设备
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"[INFO] device={DEVICE} dtype={DTYPE}")
print(f"[INFO] loading CLIP from local dir: {LOCAL_CLIP_DIR}")

# -----------------------------
# 加载 CLIP（100% 本地，不触网）
# -----------------------------
processor = CLIPProcessor.from_pretrained(str(LOCAL_CLIP_DIR), local_files_only=True)
model = CLIPModel.from_pretrained(
    str(LOCAL_CLIP_DIR),
    local_files_only=True,
    torch_dtype=DTYPE if DEVICE == "cuda" else None,
).to(DEVICE).eval()

# -----------------------------
# 读取 pairs（固定候选池）
# -----------------------------
pairs = []
with PAIRS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))
print(f"[INFO] loaded pairs: {len(pairs)}")

# -----------------------------
# A 路 decision（只是“证据标签”，不是硬规则）
# -----------------------------
def decide_A(sim: float) -> str:
    if sim >= 0.90:
        return "SUPPORT"
    if sim >= 0.80:
        return "WEAK"
    return "AGAINST"

# -----------------------------
# 主循环：逐 pair 计算 CLIP 相似度
# -----------------------------
with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
    for row in tqdm(pairs, desc="A-CLIP(L/14) on pool", ncols=100):
        try:
            q_img = Image.open(row["query_path"]).convert("RGB")
            c_img = Image.open(row["candidate_path"]).convert("RGB")

            inputs = processor(images=[q_img, c_img], return_tensors="pt")

            # move to device
            for k, v in inputs.items():
                inputs[k] = v.to(DEVICE)

            with torch.no_grad():
                feats = model.get_image_features(**inputs)  # [2, D]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim = float((feats[0] * feats[1]).sum().item())

            out = {
                "schema": "a_clip_on_pool_v1",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "query_path": row["query_path"],
                "candidate_path": row["candidate_path"],
                "clip_model_local_dir": str(LOCAL_CLIP_DIR),
                "clip_similarity": sim,
                "A_decision": decide_A(sim),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "a_clip_on_pool_v1_error",
                "query_image_id": row.get("query_image_id"),
                "candidate_image_id": row.get("candidate_image_id"),
                "query_path": row.get("query_path"),
                "candidate_path": row.get("candidate_path"),
                "error": str(e),
            }, ensure_ascii=False) + "\n")

print(f"[DONE] A-CLIP -> {OUT_FILE}")
print(f"[DONE] Errors -> {ERR_FILE}")

# %% [cell 6]
# =========================================================
# PRF 303 封板：A(rank补齐) + 30B Teacher Arbitration（冻结 Prompt）
# - Step A2: 在候选池内按 query 分组，为 A 加 rank -> a_clip_on_pool_303w_ranked.jsonl
# - Step T : 303 全异议（对每个 pair 调用 30B），v1-v4 分版本输出
#
# 输入（冻结）:
#   pairs_303w_v2.jsonl
#   a_clip_on_pool_303w.jsonl                     (已有)
#   b1_pair_review_303w_v3.jsonl                  (真实 schema: b1_pair_review_v3 + criteria.score/reason)
#   c_maskclip_303w_{v}.jsonl
#   b2_polygon_allpairs_{v}.jsonl
#
# 输出（覆盖式）:
#   checkpoints/a_clip_on_pool_303w_ranked.jsonl
#   V30Bjg/teacher_30b_303w_{v}.jsonl
#   V30Bjg/teacher_30b_303w_{v}_errors.jsonl
#   V30Bjg/teacher_30b_303w_{v}_summary.json
# =========================================================

import json, base64, time
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm

# -----------------------------
# Paths (frozen)
# -----------------------------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUTD = ROOT / "V30Bjg"
OUTD.mkdir(parents=True, exist_ok=True)

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
A_FILE     = CKPT / "a_clip_on_pool_303w.jsonl"
A_RANKED   = CKPT / "a_clip_on_pool_303w_ranked.jsonl"
B_FILE     = CKPT / "b1_pair_review_303w_v3.jsonl"

VERS = ["v1", "v2", "v3", "v3_1", "v4"]
def C_FILE(v): return CKPT / f"c_maskclip_303w_{v}.jsonl"
def P_FILE(v): return CKPT / f"b2_polygon_allpairs_{v}.jsonl"

assert PAIRS_FILE.exists(), f"Missing {PAIRS_FILE}"
assert A_FILE.exists(),     f"Missing {A_FILE}"
assert B_FILE.exists(),     f"Missing {B_FILE}"
for v in VERS:
    assert C_FILE(v).exists(), f"Missing {C_FILE(v)}"
    assert P_FILE(v).exists(), f"Missing {P_FILE(v)}"

# -----------------------------
# 30B endpoint
# -----------------------------
TEACHER_URL = "http://127.0.0.1:8081/v1/chat/completions"
HEALTH_URL  = "http://127.0.0.1:8081/health"
print("[HEALTH]", requests.get(HEALTH_URL, timeout=5).json())

# -----------------------------
# OVERWRITE
# -----------------------------
OVERWRITE = True

# =========================================================
# ✅ 冻结版 30B 仲裁 Prompt（严格按你提供的）
# =========================================================
SYSTEM_PROMPT = """
你是 PRF（Pair Review Filtering）系统中的终裁裁判（Teacher）。

在你接收到该样本之前，系统已经决定需要你参与仲裁。
你不需要、也不应该关心该样本为何被送到你这里。

A / B / C 子系统已经完成判断并提供显式证据。
你的职责不是重新评审图像，而是基于这些证据进行最终裁决，
并给出可解释的理由。

禁止：
- 重新独立评估图像
- 重新生成五维度或噪声证据
- 简单投票而不给出理由
""".strip()

USER_PROMPT_TEMPLATE = """
这是 PRF 系统中的一个 image pair。
A / B / C 已经给出了判断结果与显式证据（不同版本证据形式可能不同）。

====================
【A：CLIP 相似度证据】
- A_decision: {A_decision}
- A_score: {A_score}
- A_rank: {A_rank}

====================
【B：8B 五维度评审】
- pixel_consistency: {B_pixel}
- same_base_image: {B_base}
- same_instance: {B_instance}
- strong_structure: {B_strong}
- weak_structure: {B_weak}
- B_final: {B_decision}
- B_confidence: {B_conf}

====================
【C：Noise-aware 证据】
- noise_detected: {C_noise}
- cover_frac_raw: {C_cover_raw}
- cover_frac_capped: {C_cover_cap}
- mask_clip_raw: {C_clip_raw}
- mask_clip_masked: {C_clip_masked}
- mask_clip_delta: {C_delta}
- C_decision: {C_decision}
- C_explain: {C_explain}

====================
请基于以上证据进行仲裁，仅输出 JSON：

{
  "schema": "prf_teacher_arbitration_v1",
  "final_decision": "ACCEPT | REJECT | AMBIGUOUS",
  "adopted_from": ["A", "B", "C"],
  "overruled": ["A", "B", "C"],
  "confidence": 0.0,
  "reason": "中文一句话，说明你如何权衡证据"
}
""".strip()

# -----------------------------
# Helpers
# -----------------------------
def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suf == ".png":
        mime = "image/png"
    elif suf == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`")
        lines = t.splitlines()
        if lines and lines[0].strip().lower() == "json":
            t = "\n".join(lines[1:])
    return json.loads(t)

def schema_check_teacher(d: dict):
    assert isinstance(d, dict), "Teacher output not dict"
    assert d.get("schema") == "prf_teacher_arbitration_v1", f"Bad schema: {d.get('schema')}"
    assert d.get("final_decision") in ["ACCEPT", "REJECT", "AMBIGUOUS"], f"Bad final_decision: {d.get('final_decision')}"
    assert isinstance(d.get("adopted_from"), list), "adopted_from must be list"
    assert isinstance(d.get("overruled"), list), "overruled must be list"
    conf = d.get("confidence")
    assert isinstance(conf, (int, float)), "confidence must be number"
    assert 0.0 <= float(conf) <= 1.0, "confidence out of range"
    assert isinstance(d.get("reason"), str) and len(d["reason"].strip()) > 0, "reason must be non-empty str"

def load_jsonl_list(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_map_pair(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r, dict) and str(r.get("schema", "")).endswith("_error"):
                continue
            key = (r.get("query_image_id"), r.get("candidate_image_id"))
            m[key] = r
    return m

# -----------------------------
# STRICT B schema extractor (b1_pair_review_v3)
# -----------------------------
def pick_B_dims_strict(b: dict):
    if not b:
        return {
            "pixel_consistency": None,
            "same_base_image": None,
            "same_instance": None,
            "strong_structure": None,
            "weak_structure": None,
        }
    assert b.get("schema") in ["b1_pair_review_v3", "b1_pair_review_v3.0", "b1_pair_review_v3_1", "b1_pair_review_v3.1"], \
        f"Unexpected B schema: {b.get('schema')}"
    crit = b.get("criteria")
    assert isinstance(crit, dict), f"Bad B.criteria type: {type(crit)}"

    def score(dim):
        obj = crit.get(dim, None)
        assert isinstance(obj, dict), f"Missing/Bad criteria[{dim}]"
        s = obj.get("score", None)
        assert isinstance(s, int), f"Bad B {dim}.score (expect int): {s}"
        assert 0 <= s <= 5, f"Bad B {dim}.score out of range: {s}"
        return s

    return {
        "pixel_consistency": score("pixel_consistency"),
        "same_base_image": score("same_base_image"),
        "same_instance": score("same_instance"),
        "strong_structure": score("strong_structure"),
        "weak_structure": score("weak_structure"),
    }

def pick_B_meta_strict(b: dict):
    if not b:
        return (None, None)
    dec = b.get("final_recommendation", None)
    conf = b.get("confidence", None)
    if dec is not None:
        assert dec in ["ACCEPT", "REJECT", "AMBIGUOUS"], f"Bad B final_recommendation: {dec}"
    if conf is not None:
        assert isinstance(conf, (int, float)), f"Bad B confidence type: {type(conf)}"
        assert 0.0 <= float(conf) <= 1.0, f"Bad B confidence range: {conf}"
    return (dec, conf)

# -----------------------------
# C/P flexible pickers (keep robust because v1-v4 differ)
# -----------------------------
def pick_C_fields(c: dict):
    if not c:
        return (None, None, None, None, None)
    def g(*keys):
        for k in keys:
            if k in c:
                return c[k]
        return None
    clip_raw    = g("mask_clip_raw", "clip_raw", "raw", "similarity_raw")
    clip_masked = g("mask_clip_masked", "clip_masked", "masked", "similarity_masked")
    delta       = g("mask_clip_delta", "delta", "mask_delta", "delta_sim", "delta_similarity")
    decision    = g("C_decision", "decision", "final_decision")
    explain     = g("C_explain", "explain_zh", "reason_zh", "explain")
    return (clip_raw, clip_masked, delta, decision, explain)

def pick_P_fields(p: dict):
    if not p:
        return (None, None, None)
    noise = p.get("noise_detected", p.get("noise_decision", None))
    cover_raw = p.get("cover_frac_raw", None)
    cover_cap = p.get("cover_frac_capped", None)
    return (noise, cover_raw, cover_cap)

# -----------------------------
# Step A2: Build A_ranked file
# -----------------------------
print("\n===== STEP A2: build A rank within pool =====")
pairs_rows = load_jsonl_list(PAIRS_FILE)

# group candidate list per query from PAIRS (canonical)
by_q = defaultdict(list)
for r in pairs_rows:
    by_q[r["query_image_id"]].append(r)

A_map = load_map_pair(A_FILE)

if OVERWRITE and A_RANKED.exists():
    A_RANKED.unlink()

with A_RANKED.open("a", encoding="utf-8") as fout:
    for qid, rows in tqdm(by_q.items(), desc="A-rank", ncols=100):
        # collect sims
        tmp = []
        for r in rows:
            key = (qid, r["candidate_image_id"])
            a = A_map.get(key, None)
            sim = None if a is None else a.get("clip_similarity", None)
            tmp.append((r["candidate_image_id"], sim, a))
        # sort by sim desc (None -> -inf)
        tmp_sorted = sorted(tmp, key=lambda x: (-1e18 if x[1] is None else -float(x[1])))
        # assign rank starting at 1
        for rank_idx, (cid, sim, arow) in enumerate(tmp_sorted, start=1):
            out = {
                "schema": "a_clip_on_pool_ranked_v1",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_similarity": sim,
                "clip_rank": rank_idx,
            }
            # keep decision if existed
            if arow and "A_decision" in arow:
                out["A_decision"] = arow["A_decision"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"[DONE] A ranked -> {A_RANKED}")

A_rank_map = load_map_pair(A_RANKED)

# Load B once (strict)
B_map = load_map_pair(B_FILE)

# -----------------------------
# Teacher per version
# -----------------------------
print("\n===== STEP T: teacher arbitration per version (ALL-PAIR on 303) =====")
for v in VERS:
    print(f"\n----- VERSION {v} -----")

    C_map = load_map_pair(C_FILE(v))
    P_map = load_map_pair(P_FILE(v))

    OUT_FILE = OUTD / f"teacher_30b_303w_{v}.jsonl"
    ERR_FILE = OUTD / f"teacher_30b_303w_{v}_errors.jsonl"
    SUMM     = OUTD / f"teacher_30b_303w_{v}_summary.json"

    if OVERWRITE:
        for p in [OUT_FILE, ERR_FILE, SUMM]:
            if p.exists():
                p.unlink()

    n_total = 0
    n_ok = 0
    n_fail = 0

    with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
        for row in tqdm(pairs_rows, desc=f"30B[{v}]", ncols=100):
            n_total += 1
            qid = row["query_image_id"]
            cid = row["candidate_image_id"]
            key = (qid, cid)

            try:
                # A evidence (ranked)
                a = A_rank_map.get(key, None)
                A_decision = None if a is None else a.get("A_decision", None)
                A_score    = None if a is None else a.get("clip_similarity", None)
                A_rank     = None if a is None else a.get("clip_rank", None)

                # B evidence (strict)
                b = B_map.get(key, None)
                bd = pick_B_dims_strict(b)
                B_dec, B_conf = pick_B_meta_strict(b)

                # C + polygon evidence (version-specific)
                c = C_map.get(key, None)
                p = P_map.get(key, None)

                C_clip_raw, C_clip_masked, C_delta, C_decision, C_explain = pick_C_fields(c)
                C_noise, C_cover_raw, C_cover_cap = pick_P_fields(p)

                prompt = USER_PROMPT_TEMPLATE.format(
                    A_decision=A_decision,
                    A_score=A_score,
                    A_rank=A_rank,

                    B_pixel=bd["pixel_consistency"],
                    B_base=bd["same_base_image"],
                    B_instance=bd["same_instance"],
                    B_strong=bd["strong_structure"],
                    B_weak=bd["weak_structure"],
                    B_decision=B_dec,
                    B_conf=B_conf,

                    C_noise=C_noise,
                    C_cover_raw=C_cover_raw,
                    C_cover_cap=C_cover_cap,
                    C_clip_raw=C_clip_raw,
                    C_clip_masked=C_clip_masked,
                    C_delta=C_delta,
                    C_decision=C_decision,
                    C_explain=C_explain
                )

                payload = {
                    "model": "qwen3-vl-30b",
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["query_path"]))}},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["candidate_path"]))}},
                        ]}
                    ],
                }

                resp = requests.post(TEACHER_URL, json=payload, timeout=300)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                out = extract_json(content)
                schema_check_teacher(out)

                save = {
                    "version": v,
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    **out
                }
                fout.write(json.dumps(save, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1

            except Exception as e:
                ferr.write(json.dumps({
                    "version": v,
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                ferr.flush()
                n_fail += 1

    summary = {
        "version": v,
        "n_pairs": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "A_ranked": str(A_RANKED),
        "out": str(OUT_FILE),
        "err": str(ERR_FILE),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with SUMM.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[SUMMARY]", summary)

print("\n===== ALL DONE =====")

# %% [cell 7]
import json
from pathlib import Path
from collections import Counter

ERR = Path(r"<PROTOTYPE303_ROOT>\V30Bjg\teacher_30b_303w_v1_errors.jsonl")
assert ERR.exists()

cnt = Counter()
first = None
with ERR.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        msg = r.get("error", "")
        cnt[msg] += 1
        if first is None:
            first = r

print("[FIRST ERROR ROW]")
print(json.dumps(first, ensure_ascii=False, indent=2))
print("\n[TOP 5 ERROR TYPES]")
for k, v in cnt.most_common(5):
    print(v, "x", k[:200])

# %% [cell 8]
# =========================================================
# PRF 303 封板：A(rank补齐) + 30B Teacher Arbitration（冻结 Prompt）
# ✅ 修复点：USER_PROMPT_TEMPLATE 内 JSON 示例花括号转义 {{ }}
# =========================================================

import json, base64, time
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm

# -----------------------------
# Paths (frozen)
# -----------------------------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUTD = ROOT / "V30Bjg"
OUTD.mkdir(parents=True, exist_ok=True)

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
A_FILE     = CKPT / "a_clip_on_pool_303w.jsonl"
A_RANKED   = CKPT / "a_clip_on_pool_303w_ranked.jsonl"
B_FILE     = CKPT / "b1_pair_review_303w_v3.jsonl"

VERS = ["v1", "v2", "v3", "v3_1", "v4"]
def C_FILE(v): return CKPT / f"c_maskclip_303w_{v}.jsonl"
def P_FILE(v): return CKPT / f"b2_polygon_allpairs_{v}.jsonl"

assert PAIRS_FILE.exists()
assert A_FILE.exists()
assert B_FILE.exists()
for v in VERS:
    assert C_FILE(v).exists()
    assert P_FILE(v).exists()

# -----------------------------
# 30B endpoint
# -----------------------------
TEACHER_URL = "http://127.0.0.1:8081/v1/chat/completions"
HEALTH_URL  = "http://127.0.0.1:8081/health"
print("[HEALTH]", requests.get(HEALTH_URL, timeout=5).json())

OVERWRITE = True

# =========================================================
# ✅ 冻结版 30B 仲裁 Prompt（严格按你提供的语义；仅做花括号转义）
# =========================================================
SYSTEM_PROMPT = """
你是 PRF（Pair Review Filtering）系统中的终裁裁判（Teacher）。

在你接收到该样本之前，系统已经决定需要你参与仲裁。
你不需要、也不应该关心该样本为何被送到你这里。

A / B / C 子系统已经完成判断并提供显式证据。
你的职责不是重新评审图像，而是基于这些证据进行最终裁决，
并给出可解释的理由。

禁止：
- 重新独立评估图像
- 重新生成五维度或噪声证据
- 简单投票而不给出理由
""".strip()

# ⚠️ 关键：JSON 示例必须 {{ }} 转义，否则 .format 会 KeyError
USER_PROMPT_TEMPLATE = """
这是 PRF 系统中的一个 image pair。
A / B / C 已经给出了判断结果与显式证据（不同版本证据形式可能不同）。

====================
【A：CLIP 相似度证据】
- A_decision: {A_decision}
- A_score: {A_score}
- A_rank: {A_rank}

====================
【B：8B 五维度评审】
- pixel_consistency: {B_pixel}
- same_base_image: {B_base}
- same_instance: {B_instance}
- strong_structure: {B_strong}
- weak_structure: {B_weak}
- B_final: {B_decision}
- B_confidence: {B_conf}

====================
【C：Noise-aware 证据】
- noise_detected: {C_noise}
- cover_frac_raw: {C_cover_raw}
- cover_frac_capped: {C_cover_cap}
- mask_clip_raw: {C_clip_raw}
- mask_clip_masked: {C_clip_masked}
- mask_clip_delta: {C_delta}
- C_decision: {C_decision}
- C_explain: {C_explain}

====================
请基于以上证据进行仲裁，仅输出 JSON：

{{
  "schema": "prf_teacher_arbitration_v1",
  "final_decision": "ACCEPT | REJECT | AMBIGUOUS",
  "adopted_from": ["A", "B", "C"],
  "overruled": ["A", "B", "C"],
  "confidence": 0.0,
  "reason": "中文一句话，说明你如何权衡证据"
}}
""".strip()

# -----------------------------
# Helpers
# -----------------------------
def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suf == ".png":
        mime = "image/png"
    elif suf == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`")
        lines = t.splitlines()
        if lines and lines[0].strip().lower() == "json":
            t = "\n".join(lines[1:])
    return json.loads(t)

def schema_check_teacher(d: dict):
    assert isinstance(d, dict)
    assert d.get("schema") == "prf_teacher_arbitration_v1"
    assert d.get("final_decision") in ["ACCEPT", "REJECT", "AMBIGUOUS"]
    assert isinstance(d.get("adopted_from"), list)
    assert isinstance(d.get("overruled"), list)
    conf = d.get("confidence")
    assert isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
    assert isinstance(d.get("reason"), str) and d["reason"].strip()

def load_map_pair(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("schema","")).endswith("_error"):
                continue
            m[(r.get("query_image_id"), r.get("candidate_image_id"))] = r
    return m

def load_pairs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# STRICT B schema (b1_pair_review_v3)
def pick_B_dims_strict(b: dict):
    if not b:
        return {k: None for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]}
    crit = b["criteria"]
    def s(dim):
        v = crit[dim]["score"]
        assert isinstance(v, int) and 0 <= v <= 5
        return v
    return {
        "pixel_consistency": s("pixel_consistency"),
        "same_base_image": s("same_base_image"),
        "same_instance": s("same_instance"),
        "strong_structure": s("strong_structure"),
        "weak_structure": s("weak_structure"),
    }

def pick_B_meta_strict(b: dict):
    if not b:
        return (None, None)
    dec = b.get("final_recommendation", None)
    conf = b.get("confidence", None)
    if dec is not None:
        assert dec in ["ACCEPT","REJECT","AMBIGUOUS"]
    if conf is not None:
        assert isinstance(conf,(int,float)) and 0.0 <= float(conf) <= 1.0
    return (dec, conf)

# C/P fields (版本差异保持鲁棒)
def pick_C_fields(c: dict):
    if not c:
        return (None, None, None, None, None)
    def g(*keys):
        for k in keys:
            if k in c:
                return c[k]
        return None
    return (
        g("mask_clip_raw","clip_raw","raw"),
        g("mask_clip_masked","clip_masked","masked"),
        g("mask_clip_delta","delta","mask_delta","delta_sim"),
        g("C_decision","decision","final_decision"),
        g("C_explain","explain_zh","reason_zh","explain"),
    )

def pick_P_fields(p: dict):
    if not p:
        return (None, None, None)
    noise = p.get("noise_detected", p.get("noise_decision", None))
    return (noise, p.get("cover_frac_raw", None), p.get("cover_frac_capped", None))

# -----------------------------
# Load pool
# -----------------------------
pairs_rows = load_pairs(PAIRS_FILE)
print("[INFO] pairs:", len(pairs_rows))

by_q = defaultdict(list)
for r in pairs_rows:
    by_q[r["query_image_id"]].append(r)

# -----------------------------
# Step A2: build A rank within pool
# -----------------------------
print("\n===== STEP A2: build A rank within pool =====")
A_map_raw = load_map_pair(A_FILE)

if OVERWRITE and A_RANKED.exists():
    A_RANKED.unlink()

with A_RANKED.open("a", encoding="utf-8") as fout:
    for qid, rows in tqdm(by_q.items(), desc="A-rank", ncols=100):
        tmp = []
        for r in rows:
            key = (qid, r["candidate_image_id"])
            a = A_map_raw.get(key)
            sim = None if a is None else a.get("clip_similarity")
            tmp.append((r["candidate_image_id"], sim, a))
        tmp_sorted = sorted(tmp, key=lambda x: (-float(x[1]) if x[1] is not None else float("inf")))
        for rank_idx, (cid, sim, arow) in enumerate(tmp_sorted, start=1):
            out = {
                "schema": "a_clip_on_pool_ranked_v1",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_similarity": sim,
                "clip_rank": rank_idx,
                "A_decision": None if not arow else arow.get("A_decision"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print("[DONE] A ranked ->", A_RANKED)

A_rank_map = load_map_pair(A_RANKED)
B_map = load_map_pair(B_FILE)

# -----------------------------
# Teacher per version
# -----------------------------
print("\n===== STEP T: teacher arbitration per version (ALL-PAIR on 303) =====")

for v in VERS:
    print(f"\n----- VERSION {v} -----")

    C_map = load_map_pair(C_FILE(v))
    P_map = load_map_pair(P_FILE(v))

    OUT_FILE = OUTD / f"teacher_30b_303w_{v}.jsonl"
    ERR_FILE = OUTD / f"teacher_30b_303w_{v}_errors.jsonl"
    SUMM     = OUTD / f"teacher_30b_303w_{v}_summary.json"

    if OVERWRITE:
        for p in [OUT_FILE, ERR_FILE, SUMM]:
            if p.exists():
                p.unlink()

    n_total = 0
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
        for row in tqdm(pairs_rows, desc=f"30B[{v}]", ncols=100):
            n_total += 1
            qid = row["query_image_id"]
            cid = row["candidate_image_id"]
            key = (qid, cid)

            try:
                a = A_rank_map.get(key)
                b = B_map.get(key)
                c = C_map.get(key)
                p = P_map.get(key)

                A_decision = None if not a else a.get("A_decision")
                A_score    = None if not a else a.get("clip_similarity")
                A_rank     = None if not a else a.get("clip_rank")

                bd = pick_B_dims_strict(b)
                B_dec, B_conf = pick_B_meta_strict(b)

                C_raw, C_mask, C_delta, C_decision, C_explain = pick_C_fields(c)
                C_noise, C_cover_raw, C_cover_cap = pick_P_fields(p)

                prompt = USER_PROMPT_TEMPLATE.format(
                    A_decision=A_decision,
                    A_score=A_score,
                    A_rank=A_rank,

                    B_pixel=bd["pixel_consistency"],
                    B_base=bd["same_base_image"],
                    B_instance=bd["same_instance"],
                    B_strong=bd["strong_structure"],
                    B_weak=bd["weak_structure"],
                    B_decision=B_dec,
                    B_conf=B_conf,

                    C_noise=C_noise,
                    C_cover_raw=C_cover_raw,
                    C_cover_cap=C_cover_cap,
                    C_clip_raw=C_raw,
                    C_clip_masked=C_mask,
                    C_delta=C_delta,
                    C_decision=C_decision,
                    C_explain=C_explain
                )

                payload = {
                    "model": "qwen3-vl-30b",
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["query_path"]))}},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["candidate_path"]))}},
                        ]}
                    ],
                }

                resp = requests.post(TEACHER_URL, json=payload, timeout=300)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                out = extract_json(content)
                schema_check_teacher(out)

                save = {
                    "version": v,
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    **out
                }
                fout.write(json.dumps(save, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1

            except Exception as e:
                ferr.write(json.dumps({
                    "version": v,
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                ferr.flush()
                n_fail += 1

    summary = {
        "version": v,
        "n_pairs": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "A_ranked": str(A_RANKED),
        "out": str(OUT_FILE),
        "err": str(ERR_FILE),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with SUMM.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[SUMMARY]", summary)

print("\n===== ALL DONE =====")

# %% [cell 9]
# =========================================================
# PRF 303 封板：A(rank补齐) + 30B Teacher Arbitration（冻结 Prompt）全版本跑完
# 输出：
#   checkpoints/a_clip_on_pool_303w_ranked.jsonl
#   V30Bjg/teacher_30b_303w_{v}.jsonl
#   V30Bjg/teacher_30b_303w_{v}_errors.jsonl
#   V30Bjg/teacher_30b_303w_{v}_summary.json
# =========================================================

import json, base64, time
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm

# -----------------------------
# Paths (frozen)
# -----------------------------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUTD = ROOT / "V30Bjg"
OUTD.mkdir(parents=True, exist_ok=True)

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
A_FILE     = CKPT / "a_clip_on_pool_303w.jsonl"
A_RANKED   = CKPT / "a_clip_on_pool_303w_ranked.jsonl"
B_FILE     = CKPT / "b1_pair_review_303w_v3.jsonl"

VERS = ["v1", "v2", "v3", "v3_1", "v4"]
def C_FILE(v): return CKPT / f"c_maskclip_303w_{v}.jsonl"
def P_FILE(v): return CKPT / f"b2_polygon_allpairs_{v}.jsonl"

# -----------------------------
# Controls
# -----------------------------
OVERWRITE = True          # True: 覆盖式跑（推荐封板）
RESUME_IF_EXISTS = False  # True: 断点续跑（若OVERWRITE=False时有意义）
PRINT_EVERY = 50          # 每50条打印一次进度（确认不是秒出假跑）
TIMEOUT_SEC = 300         # 单请求超时（秒）

# -----------------------------
# 30B endpoint
# -----------------------------
TEACHER_URL = "http://127.0.0.1:8081/v1/chat/completions"
HEALTH_URL  = "http://127.0.0.1:8081/health"
print("[HEALTH]", requests.get(HEALTH_URL, timeout=5).json())

# -----------------------------
# Validate inputs
# -----------------------------
assert PAIRS_FILE.exists(), f"Missing {PAIRS_FILE}"
assert A_FILE.exists(), f"Missing {A_FILE}"
assert B_FILE.exists(), f"Missing {B_FILE}"
for v in VERS:
    assert C_FILE(v).exists(), f"Missing {C_FILE(v)}"
    assert P_FILE(v).exists(), f"Missing {P_FILE(v)}"

# =========================================================
# ✅ 冻结版 30B 仲裁 Prompt（严格按你提供的语义；仅做花括号转义）
# =========================================================
SYSTEM_PROMPT = """
你是 PRF（Pair Review Filtering）系统中的终裁裁判（Teacher）。

在你接收到该样本之前，系统已经决定需要你参与仲裁。
你不需要、也不应该关心该样本为何被送到你这里。

A / B / C 子系统已经完成判断并提供显式证据。
你的职责不是重新评审图像，而是基于这些证据进行最终裁决，
并给出可解释的理由。

禁止：
- 重新独立评估图像
- 重新生成五维度或噪声证据
- 简单投票而不给出理由
""".strip()

# ⚠️ JSON 示例必须 {{ }} 转义，否则 .format 会 KeyError
USER_PROMPT_TEMPLATE = """
这是 PRF 系统中的一个 image pair。
A / B / C 已经给出了判断结果与显式证据（不同版本证据形式可能不同）。

====================
【A：CLIP 相似度证据】
- A_decision: {A_decision}
- A_score: {A_score}
- A_rank: {A_rank}

====================
【B：8B 五维度评审】
- pixel_consistency: {B_pixel}
- same_base_image: {B_base}
- same_instance: {B_instance}
- strong_structure: {B_strong}
- weak_structure: {B_weak}
- B_final: {B_decision}
- B_confidence: {B_conf}

====================
【C：Noise-aware 证据】
- noise_detected: {C_noise}
- cover_frac_raw: {C_cover_raw}
- cover_frac_capped: {C_cover_cap}
- mask_clip_raw: {C_clip_raw}
- mask_clip_masked: {C_clip_masked}
- mask_clip_delta: {C_delta}
- C_decision: {C_decision}
- C_explain: {C_explain}

====================
请基于以上证据进行仲裁，仅输出 JSON：

{{
  "schema": "prf_teacher_arbitration_v1",
  "final_decision": "ACCEPT | REJECT | AMBIGUOUS",
  "adopted_from": ["A", "B", "C"],
  "overruled": ["A", "B", "C"],
  "confidence": 0.0,
  "reason": "中文一句话，说明你如何权衡证据"
}}
""".strip()

# -----------------------------
# Helpers
# -----------------------------
def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suf == ".png":
        mime = "image/png"
    elif suf == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`")
        lines = t.splitlines()
        if lines and lines[0].strip().lower() == "json":
            t = "\n".join(lines[1:])
    return json.loads(t)

def schema_check_teacher(d: dict):
    assert isinstance(d, dict)
    assert d.get("schema") == "prf_teacher_arbitration_v1"
    assert d.get("final_decision") in ["ACCEPT", "REJECT", "AMBIGUOUS"]
    assert isinstance(d.get("adopted_from"), list)
    assert isinstance(d.get("overruled"), list)
    conf = d.get("confidence")
    assert isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
    assert isinstance(d.get("reason"), str) and d["reason"].strip()

def load_pairs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_map_pair(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("schema","")).endswith("_error"):
                continue
            m[(r.get("query_image_id"), r.get("candidate_image_id"))] = r
    return m

# STRICT B schema (b1_pair_review_v3)
def pick_B_dims_strict(b: dict):
    if not b:
        return {k: None for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]}
    crit = b["criteria"]
    def s(dim):
        v = crit[dim]["score"]
        assert isinstance(v, int) and 0 <= v <= 5
        return v
    return {
        "pixel_consistency": s("pixel_consistency"),
        "same_base_image": s("same_base_image"),
        "same_instance": s("same_instance"),
        "strong_structure": s("strong_structure"),
        "weak_structure": s("weak_structure"),
    }

def pick_B_meta_strict(b: dict):
    if not b:
        return (None, None)
    dec = b.get("final_recommendation", None)
    conf = b.get("confidence", None)
    if dec is not None:
        assert dec in ["ACCEPT","REJECT","AMBIGUOUS"]
    if conf is not None:
        assert isinstance(conf,(int,float)) and 0.0 <= float(conf) <= 1.0
    return (dec, conf)

# C/P fields (v1-v4差异保持鲁棒)
def pick_C_fields(c: dict):
    if not c:
        return (None, None, None, None, None)
    def g(*keys):
        for k in keys:
            if k in c:
                return c[k]
        return None
    return (
        g("mask_clip_raw","clip_raw","raw"),
        g("mask_clip_masked","clip_masked","masked"),
        g("mask_clip_delta","delta","mask_delta","delta_sim"),
        g("C_decision","decision","final_decision"),
        g("C_explain","explain_zh","reason_zh","explain"),
    )

def pick_P_fields(p: dict):
    if not p:
        return (None, None, None)
    noise = p.get("noise_detected", p.get("noise_decision", None))
    return (noise, p.get("cover_frac_raw", None), p.get("cover_frac_capped", None))

# -----------------------------
# Load pool
# -----------------------------
pairs_rows = load_pairs(PAIRS_FILE)
print("[INFO] pairs:", len(pairs_rows))

# group by query for rank
by_q = defaultdict(list)
for r in pairs_rows:
    by_q[r["query_image_id"]].append(r)

# -----------------------------
# Step A2: build A rank within pool
# -----------------------------
print("\n===== STEP A2: build A rank within pool =====")
A_map_raw = load_map_pair(A_FILE)

if OVERWRITE and A_RANKED.exists():
    A_RANKED.unlink()

with A_RANKED.open("a", encoding="utf-8") as fout:
    for qid, rows in tqdm(by_q.items(), desc="A-rank", ncols=100):
        tmp = []
        for r in rows:
            key = (qid, r["candidate_image_id"])
            a = A_map_raw.get(key)
            sim = None if a is None else a.get("clip_similarity")
            tmp.append((r["candidate_image_id"], sim, a))
        # sort by sim desc (None last)
        tmp_sorted = sorted(tmp, key=lambda x: (-float(x[1]) if x[1] is not None else float("-inf")))
        for rank_idx, (cid, sim, arow) in enumerate(tmp_sorted, start=1):
            out = {
                "schema": "a_clip_on_pool_ranked_v1",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_similarity": sim,
                "clip_rank": rank_idx,
                "A_decision": None if not arow else arow.get("A_decision"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print("[DONE] A ranked ->", A_RANKED)

A_rank_map = load_map_pair(A_RANKED)
B_map = load_map_pair(B_FILE)

# -----------------------------
# Teacher runner per version
# -----------------------------
print("\n===== STEP T: teacher arbitration per version (ALL-PAIR on 303) =====")

for v in VERS:
    print(f"\n----- VERSION {v} -----")

    C_map = load_map_pair(C_FILE(v))
    P_map = load_map_pair(P_FILE(v))

    OUT_FILE = OUTD / f"teacher_30b_303w_{v}.jsonl"
    ERR_FILE = OUTD / f"teacher_30b_303w_{v}_errors.jsonl"
    SUMM     = OUTD / f"teacher_30b_303w_{v}_summary.json"

    if OVERWRITE:
        for p in [OUT_FILE, ERR_FILE, SUMM]:
            if p.exists():
                p.unlink()

    done = set()
    if (not OVERWRITE) and RESUME_IF_EXISTS and OUT_FILE.exists():
        with OUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r.get("query_image_id"), r.get("candidate_image_id")))
        print("[RESUME] done:", len(done))

    n_total = 0
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
        for i, row in enumerate(tqdm(pairs_rows, desc=f"30B[{v}]", ncols=100), start=1):
            qid = row["query_image_id"]
            cid = row["candidate_image_id"]
            key = (qid, cid)

            if key in done:
                continue

            n_total += 1

            try:
                a = A_rank_map.get(key)
                b = B_map.get(key)
                c = C_map.get(key)
                p = P_map.get(key)

                A_decision = None if not a else a.get("A_decision")
                A_score    = None if not a else a.get("clip_similarity")
                A_rank     = None if not a else a.get("clip_rank")

                bd = pick_B_dims_strict(b)
                B_dec, B_conf = pick_B_meta_strict(b)

                C_raw, C_mask, C_delta, C_decision, C_explain = pick_C_fields(c)
                C_noise, C_cover_raw, C_cover_cap = pick_P_fields(p)

                prompt = USER_PROMPT_TEMPLATE.format(
                    A_decision=A_decision,
                    A_score=A_score,
                    A_rank=A_rank,

                    B_pixel=bd["pixel_consistency"],
                    B_base=bd["same_base_image"],
                    B_instance=bd["same_instance"],
                    B_strong=bd["strong_structure"],
                    B_weak=bd["weak_structure"],
                    B_decision=B_dec,
                    B_conf=B_conf,

                    C_noise=C_noise,
                    C_cover_raw=C_cover_raw,
                    C_cover_cap=C_cover_cap,
                    C_clip_raw=C_raw,
                    C_clip_masked=C_mask,
                    C_delta=C_delta,
                    C_decision=C_decision,
                    C_explain=C_explain
                )

                payload = {
                    "model": "qwen3-vl-30b",
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["query_path"]))}},
                            {"type": "image_url", "image_url": {"url": to_data_url(Path(row["candidate_path"]))}},
                        ]}
                    ],
                }

                resp = requests.post(TEACHER_URL, json=payload, timeout=TIMEOUT_SEC)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                out = extract_json(content)
                schema_check_teacher(out)

                save = {"version": v, "query_image_id": qid, "candidate_image_id": cid, **out}
                fout.write(json.dumps(save, ensure_ascii=False) + "\n")
                fout.flush()

                n_ok += 1
                done.add(key)

            except Exception as e:
                ferr.write(json.dumps({
                    "version": v,
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                ferr.flush()
                n_fail += 1

            if i % PRINT_EVERY == 0:
                dt = time.time() - t0
                print(f"[PROGRESS {v}] i={i} ok={n_ok} fail={n_fail} elapsed={dt:.1f}s")

    summary = {
        "version": v,
        "n_pairs_total": len(pairs_rows),
        "n_processed_this_run": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "A_ranked": str(A_RANKED),
        "out": str(OUT_FILE),
        "err": str(ERR_FILE),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with SUMM.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[SUMMARY]", summary)

print("\n===== ALL DONE =====")

# %% [cell 10]
# =========================================================
# PRF 303 封板：30B Teacher 全分析（一次性跑完）
# 输出目录：<PROTOTYPE303_ROOT>\V30Bjg\30BFX
# 读取：
#   V30Bjg/teacher_30b_303w_v{v}.jsonl
#   outputs/table_3criteria_*.csv（优先 *_withcap.csv）
# 产出（全部落盘到 30BFX）：
#   - teacher_long.csv / teacher_wide.csv
#   - teacher_decision_dist.csv
#   - teacher_adopted_from_dist.csv / teacher_overruled_dist.csv
#   - teacher_confidence_stats.csv
#   - teacher_pairwise_agreement.csv / teacher_pairwise_kappa.csv
#   - teacher_all_versions_agree.csv
#   - merged_3criteria_plus_teacher.csv（把三验证表 + teacher 统计合并）
#   - summary_teacher_analysis.json（总摘要）
# =========================================================

import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
V30B = ROOT / "V30Bjg"
OUT  = V30B / "30BFX"
OUT.mkdir(parents=True, exist_ok=True)

VERS = ["v1", "v2", "v3", "v3_1", "v4"]

def teacher_file(v):
    return V30B / f"teacher_30b_303w_{v}.jsonl"

# pick 3-criteria table (prefer withcap)
CAND_TABLES = [
    ROOT / "outputs" / "table_3criteria_v1_v2_v3_v3_1_v4_withcap.csv",
    ROOT / "outputs" / "table_3criteria_v1_v2_v3_v3_1_v4.csv",
    ROOT / "outputs" / "table_3criteria_v1_v2_v3_v3_1_v4_withcap.xlsx",
    ROOT / "outputs" / "table_3criteria_v1_v2_v3_v3_1_v4.xlsx",
]
CRIT_PATH = next((p for p in CAND_TABLES if p.exists()), None)

print("[INFO] OUT:", OUT)
print("[INFO] criteria table:", CRIT_PATH if CRIT_PATH else "NOT FOUND (will still run teacher-only analysis)")

# -----------------------------
# Helpers
# -----------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def normalize_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # sometimes a single string
    return [x]

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def cohen_kappa(labels_a, labels_b, classes):
    """
    Cohen's kappa for two label arrays over same index, with fixed class set.
    """
    a = np.array(labels_a, dtype=object)
    b = np.array(labels_b, dtype=object)

    # confusion
    idx = {c:i for i,c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for x,y in zip(a,b):
        if x in idx and y in idx:
            m[idx[x], idx[y]] += 1

    n = m.sum()
    if n == 0:
        return np.nan

    po = np.trace(m) / n
    pe = (m.sum(axis=1) / n * m.sum(axis=0) / n).sum()
    if pe >= 1.0:
        return np.nan
    return (po - pe) / (1 - pe)

# -----------------------------
# Load teacher JSONL -> long table
# -----------------------------
rows = []
for v in VERS:
    fp = teacher_file(v)
    assert fp.exists(), f"Missing teacher file: {fp}"
    n = 0
    for r in iter_jsonl(fp):
        n += 1
        rows.append({
            "version": v,
            "query_image_id": r.get("query_image_id"),
            "candidate_image_id": r.get("candidate_image_id"),
            "final_decision": r.get("final_decision"),
            "confidence": safe_float(r.get("confidence")),
            "adopted_from": "|".join(normalize_list(r.get("adopted_from"))),
            "overruled": "|".join(normalize_list(r.get("overruled"))),
            "reason": r.get("reason"),
            "schema": r.get("schema"),
        })
    print(f"[INFO] loaded {v}: {n} rows")

df_long = pd.DataFrame(rows)
assert df_long["final_decision"].isin(["ACCEPT","REJECT","AMBIGUOUS"]).all(), "Unexpected final_decision values found"
assert (df_long["schema"] == "prf_teacher_arbitration_v1").all(), "Unexpected teacher schema found"

# basic sanity
n_pairs_by_v = df_long.groupby("version")[["query_image_id","candidate_image_id"]].nunique()
print("[INFO] unique keys per version:\n", n_pairs_by_v)

# -----------------------------
# Save long
# -----------------------------
long_path = OUT / "teacher_long.csv"
df_long.to_csv(long_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_long.csv ->", long_path)

# -----------------------------
# Build wide decision/conf/adopt/overruled
# -----------------------------
key_cols = ["query_image_id", "candidate_image_id"]
pivot_dec = df_long.pivot_table(index=key_cols, columns="version", values="final_decision", aggfunc="first")
pivot_conf = df_long.pivot_table(index=key_cols, columns="version", values="confidence", aggfunc="first")
pivot_adopt = df_long.pivot_table(index=key_cols, columns="version", values="adopted_from", aggfunc="first")
pivot_over = df_long.pivot_table(index=key_cols, columns="version", values="overruled", aggfunc="first")

# flatten columns
pivot_dec.columns  = [f"{c}_decision" for c in pivot_dec.columns]
pivot_conf.columns = [f"{c}_confidence" for c in pivot_conf.columns]
pivot_adopt.columns= [f"{c}_adopted_from" for c in pivot_adopt.columns]
pivot_over.columns = [f"{c}_overruled" for c in pivot_over.columns]

df_wide = pd.concat([pivot_dec, pivot_conf, pivot_adopt, pivot_over], axis=1).reset_index()

wide_path = OUT / "teacher_wide.csv"
df_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_wide.csv ->", wide_path)

# -----------------------------
# Decision distributions per version
# -----------------------------
dist = (
    df_long.groupby(["version","final_decision"])
    .size()
    .reset_index(name="count")
)
tot = df_long.groupby("version").size().reset_index(name="total")
dist = dist.merge(tot, on="version", how="left")
dist["share"] = dist["count"] / dist["total"]
dist_path = OUT / "teacher_decision_dist.csv"
dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_decision_dist.csv ->", dist_path)

# -----------------------------
# Adopted_from / Overruled distributions (tokenized)
# -----------------------------
def explode_tokens(series, sep="|"):
    out = []
    for s in series.fillna("").astype(str):
        toks = [t for t in s.split(sep) if t]
        if not toks:
            out.append(["(empty)"])
        else:
            out.append(toks)
    return out

def token_dist(df, colname, outname):
    recs = []
    for v in VERS:
        sub = df[df["version"] == v]
        toks_list = explode_tokens(sub[colname])
        c = Counter()
        for toks in toks_list:
            for t in toks:
                c[t] += 1
        total = len(sub)
        for k, cnt in c.most_common():
            recs.append({"version": v, "token": k, "count": cnt, "share": cnt/total})
    outdf = pd.DataFrame(recs)
    out_path = OUT / outname
    outdf.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[DONE]", outname, "->", out_path)
    return outdf

adopt_df = token_dist(df_long, "adopted_from", "teacher_adopted_from_dist.csv")
over_df  = token_dist(df_long, "overruled", "teacher_overruled_dist.csv")

# -----------------------------
# Confidence stats per version & by decision
# -----------------------------
conf_stats = (
    df_long.groupby(["version"])
    .agg(
        n=("confidence","count"),
        mean=("confidence","mean"),
        std=("confidence","std"),
        p10=("confidence", lambda x: np.nanpercentile(x, 10)),
        p50=("confidence", lambda x: np.nanpercentile(x, 50)),
        p90=("confidence", lambda x: np.nanpercentile(x, 90)),
        min=("confidence","min"),
        max=("confidence","max"),
    )
    .reset_index()
)
conf_stats_path = OUT / "teacher_confidence_stats.csv"
conf_stats.to_csv(conf_stats_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_confidence_stats.csv ->", conf_stats_path)

conf_by_dec = (
    df_long.groupby(["version","final_decision"])
    .agg(
        n=("confidence","count"),
        mean=("confidence","mean"),
        std=("confidence","std"),
        p50=("confidence", lambda x: np.nanpercentile(x, 50)),
        p90=("confidence", lambda x: np.nanpercentile(x, 90)),
    )
    .reset_index()
)
conf_by_dec_path = OUT / "teacher_confidence_by_decision.csv"
conf_by_dec.to_csv(conf_by_dec_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_confidence_by_decision.csv ->", conf_by_dec_path)

# -----------------------------
# Pairwise agreement + Cohen's kappa
# -----------------------------
classes = ["ACCEPT","REJECT","AMBIGUOUS"]

# build aligned matrix of decisions per version
mat = pd.DataFrame(index=df_wide.set_index(key_cols).index)
for v in VERS:
    mat[v] = df_wide.set_index(key_cols)[f"{v}_decision"]

agree_rows = []
kappa_rows = []

for i, va in enumerate(VERS):
    for vb in VERS[i+1:]:
        a = mat[va].values
        b = mat[vb].values
        agree = (a == b).mean()
        kap = cohen_kappa(a, b, classes)
        agree_rows.append({"a": va, "b": vb, "agreement": float(agree)})
        kappa_rows.append({"a": va, "b": vb, "kappa": float(kap) if kap==kap else np.nan})

df_agree = pd.DataFrame(agree_rows).sort_values(["a","b"])
df_kappa = pd.DataFrame(kappa_rows).sort_values(["a","b"])

agree_path = OUT / "teacher_pairwise_agreement.csv"
kappa_path = OUT / "teacher_pairwise_kappa.csv"
df_agree.to_csv(agree_path, index=False, encoding="utf-8-sig")
df_kappa.to_csv(kappa_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_pairwise_agreement.csv ->", agree_path)
print("[DONE] teacher_pairwise_kappa.csv ->", kappa_path)

# -----------------------------
# All-versions agreement (how many pairs have identical decision across all versions)
# -----------------------------
all_equal = mat.nunique(axis=1) == 1
df_all = pd.DataFrame({
    "all_versions_agree": all_equal.values
})
df_all_path = OUT / "teacher_all_versions_agree.csv"
df_all.to_csv(df_all_path, index=False, encoding="utf-8-sig")
print("[DONE] teacher_all_versions_agree.csv ->", df_all_path)

all_agree_rate = float(all_equal.mean())
print("[INFO] all-versions agree rate:", all_agree_rate)

# -----------------------------
# Merge teacher summary into 3-criteria table (if exists)
# We create per-version teacher summary: decision shares + adopted_from shares (A/B/C) + mean confidence
# then join on 'version'
# -----------------------------
merged_path = None
df_crit = None
if CRIT_PATH:
    if CRIT_PATH.suffix.lower() in [".csv"]:
        df_crit = pd.read_csv(CRIT_PATH)
    else:
        # xlsx
        df_crit = pd.read_excel(CRIT_PATH)

    # normalize version col
    if "version" not in df_crit.columns:
        raise RuntimeError(f"criteria table has no 'version' column: {CRIT_PATH}")

    # per-version decision share table
    dec_pivot = dist.pivot_table(index="version", columns="final_decision", values="share", aggfunc="first").reset_index()
    dec_pivot.columns = ["version"] + [f"teacher_share_{c.lower()}" for c in dec_pivot.columns[1:]]

    # adopted_from token shares (A/B/C + (empty))
    adopt_p = adopt_df.pivot_table(index="version", columns="token", values="share", aggfunc="first").reset_index()
    # prefix columns
    adopt_p.columns = ["version"] + [f"teacher_adopt_share_{c}" for c in adopt_p.columns[1:]]

    # overruled token shares
    over_p = over_df.pivot_table(index="version", columns="token", values="share", aggfunc="first").reset_index()
    over_p.columns = ["version"] + [f"teacher_overrule_share_{c}" for c in over_p.columns[1:]]

    # join summaries
    df_teacher_sum = conf_stats[["version","mean","p50","p90"]].rename(columns={
        "mean":"teacher_conf_mean",
        "p50":"teacher_conf_p50",
        "p90":"teacher_conf_p90"
    })
    df_teacher_sum = df_teacher_sum.merge(dec_pivot, on="version", how="left")
    df_teacher_sum = df_teacher_sum.merge(adopt_p, on="version", how="left")
    df_teacher_sum = df_teacher_sum.merge(over_p, on="version", how="left")

    # add global agreement scalar as a constant column (same for all rows; useful in summary)
    df_teacher_sum["teacher_all_versions_agree_rate"] = all_agree_rate

    df_merged = df_crit.merge(df_teacher_sum, on="version", how="left")
    merged_path = OUT / "merged_3criteria_plus_teacher.csv"
    df_merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print("[DONE] merged_3criteria_plus_teacher.csv ->", merged_path)

# -----------------------------
# Global summary JSON
# -----------------------------
summary = {
    "out_dir": str(OUT),
    "teacher_files": {v: str(teacher_file(v)) for v in VERS},
    "n_pairs_each_version": {v: int(df_long[df_long["version"]==v].shape[0]) for v in VERS},
    "decision_dist": {
        v: {
            d: float(dist[(dist["version"]==v) & (dist["final_decision"]==d)]["share"].iloc[0])
            if ((dist["version"]==v) & (dist["final_decision"]==d)).any() else 0.0
            for d in ["ACCEPT","REJECT","AMBIGUOUS"]
        } for v in VERS
    },
    "confidence_mean": {v: float(conf_stats[conf_stats["version"]==v]["mean"].iloc[0]) for v in VERS},
    "pairwise_agreement": df_agree.to_dict(orient="records"),
    "pairwise_kappa": df_kappa.to_dict(orient="records"),
    "all_versions_agree_rate": all_agree_rate,
    "criteria_table_used": str(CRIT_PATH) if CRIT_PATH else None,
    "merged_table_path": str(merged_path) if merged_path else None,
}

sum_path = OUT / "summary_teacher_analysis.json"
with sum_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[DONE] summary_teacher_analysis.json ->", sum_path)

print("\n===== ALL ANALYSIS DONE =====")
print("Folder:", OUT)

# %% [cell 11]
import pandas as pd
import json
from pathlib import Path
import random

# ============================
# 配置
# ============================
ROOT = Path(r"<PROTOTYPE303_ROOT>\V30Bjg\30BFX")
OUT  = ROOT / "samples"
OUT.mkdir(parents=True, exist_ok=True)

N_DISPUTE = 20   # 分歧样本数
N_RANDOM  = 20   # 每个版本随机样本数

VERS = ["v1", "v2", "v3", "v3_1", "v4"]

# ============================
# 读取 Teacher 宽表 & 长表
# ============================
df_wide = pd.read_csv(ROOT / "teacher_wide.csv")
df_long = pd.read_csv(ROOT / "teacher_long.csv")

KEYS = ["query_image_id", "candidate_image_id"]

# ============================
# A️⃣ 分歧子集
# ============================
decision_cols = [f"{v}_decision" for v in VERS]

df_wide["n_unique_decisions"] = df_wide[decision_cols].nunique(axis=1)
df_dispute = df_wide[df_wide["n_unique_decisions"] > 1]

print(f"[INFO] total dispute pairs: {len(df_dispute)}")

df_dispute_sample = df_dispute.sample(
    n=min(N_DISPUTE, len(df_dispute)),
    random_state=42
)

# ---- 保存 CSV（快速看）
csv_dispute = OUT / "sample_dispute_pairs.csv"
df_dispute_sample.to_csv(csv_dispute, index=False, encoding="utf-8-sig")
print("[DONE]", csv_dispute)

# ---- 保存 JSON（完整证据）
json_dispute = []
for _, row in df_dispute_sample.iterrows():
    q, c = row["query_image_id"], row["candidate_image_id"]
    records = df_long[
        (df_long["query_image_id"] == q) &
        (df_long["candidate_image_id"] == c)
    ].to_dict(orient="records")
    json_dispute.append({
        "query_image_id": q,
        "candidate_image_id": c,
        "teacher_records": records
    })

json_dispute_path = OUT / "sample_dispute_pairs.json"
with open(json_dispute_path, "w", encoding="utf-8") as f:
    json.dump(json_dispute, f, ensure_ascii=False, indent=2)

print("[DONE]", json_dispute_path)

# ============================
# B️⃣ 各版本随机样本（对照）
# ============================
random_samples = {}

for v in VERS:
    df_v = df_long[df_long["version"] == v]
    df_v_sample = df_v.sample(
        n=min(N_RANDOM, len(df_v)),
        random_state=42
    )
    random_samples[v] = df_v_sample

    # CSV
    csv_path = OUT / f"sample_random_{v}.csv"
    df_v_sample.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("[DONE]", csv_path)

    # JSON
    json_path = OUT / f"sample_random_{v}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            df_v_sample.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2
        )
    print("[DONE]", json_path)

# ============================
# 汇总说明
# ============================
print("\n===== SAMPLE EXTRACTION DONE =====")
print("Folder:", OUT)
print(f"- dispute samples: {len(df_dispute_sample)}")
print(f"- random per version: {N_RANDOM}")

# %% [cell 12]

