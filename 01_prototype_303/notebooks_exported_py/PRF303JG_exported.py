# AUTO-EXPORTED FROM ORIGINAL JUPYTER NOTEBOOK
# This file is DERIVED and was not executed during export.
# Source notebook: PRF303JG.ipynb

# %% [cell 1]
# =========================================================
# PRF 303 封板：分歧子集抽样 + Case 包（可解释性审计）
# 输出目录（你指定的）：
#   <PROTOTYPE303_ROOT>\V30Bjg\30Bfenqiziji
#
# 产物：
#   - manifest_dispute.csv / manifest_random.csv
#   - cases/dispute/<case_id>/{case.json, case.html}
#   - cases/random/<v>/<case_id>/{case.json, case.html}
#   - summary.json（统计：分歧规模、按版本对比等）
#
# 读取（只读，不改原始结果）：
#   - V30Bjg/teacher_30b_303w_v*.jsonl
#   - 303test/checkpoints/pairs_303w_v2.jsonl
#   - 303test/checkpoints/a_clip_on_pool_303w_ranked.jsonl (若不存在则回退 a_clip_on_pool_303w.jsonl)
#   - 303test/checkpoints/b1_pair_review_303w_v3.jsonl
#   - 303test/checkpoints/c_maskclip_303w_v*.jsonl
#   - 303test/checkpoints/b2_polygon_allpairs_v*.jsonl
# =========================================================

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
ROOT_303 = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT_303 / "checkpoints"
V30B = ROOT_303 / "V30Bjg"

OUT = Path(r"<PROTOTYPE303_ROOT>\V30Bjg\30Bfenqiziji")
OUT.mkdir(parents=True, exist_ok=True)

VERS = ["v1", "v2", "v3", "v3_1", "v4"]

# sampling
SEED = 42
N_DISPUTE = 30            # 分歧子集抽样数量（可改）
N_RANDOM_PER_VER = 20     # 每个版本随机抽样（可改）

OVERWRITE = True          # 覆盖输出（推荐封板阶段）

# inputs
PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
A_RANKED = CKPT / "a_clip_on_pool_303w_ranked.jsonl"
A_RAW    = CKPT / "a_clip_on_pool_303w.jsonl"
B_FILE   = CKPT / "b1_pair_review_303w_v3.jsonl"

def C_FILE(v): return CKPT / f"c_maskclip_303w_{v}.jsonl"
def P_FILE(v): return CKPT / f"b2_polygon_allpairs_{v}.jsonl"
def T_FILE(v): return V30B / f"teacher_30b_303w_{v}.jsonl"

# optional viz (如果存在就关联到 case，缺失不影响封板)
VIZ_PAIRS_DIR = ROOT_303 / "viz" / "pairs"

# -----------------------------
# IO helpers
# -----------------------------
def ensure_clean_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    if OVERWRITE:
        # 清空目录（只清 OUT 下的，不碰别的）
        for item in p.glob("*"):
            if item.is_file():
                item.unlink()
            else:
                # 递归删
                for sub in item.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted([x for x in item.rglob("*") if x.is_dir()], reverse=True):
                    try: sub.rmdir()
                    except: pass
                try: item.rmdir()
                except: pass

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_map_pair(path: Path):
    m = {}
    if not path.exists():
        return m
    for r in iter_jsonl(path):
        if str(r.get("schema","")).endswith("_error"):
            continue
        k = (r.get("query_image_id"), r.get("candidate_image_id"))
        m[k] = r
    return m

def safe_get(d, *keys, default=None):
    for k in keys:
        if d is not None and k in d:
            return d[k]
    return default

def b_dims_strict(b: dict):
    # schema 已知：criteria.{dim}.{score,reason}
    if not b:
        return None
    crit = b.get("criteria", {})
    out = {}
    for dim in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        if dim in crit and isinstance(crit[dim], dict):
            out[dim] = {"score": crit[dim].get("score"), "reason": crit[dim].get("reason")}
        else:
            out[dim] = {"score": None, "reason": None}
    return out

def find_viz_dir_for_pair(qid: str, cid: str):
    """
    尝试在 viz/pairs 下找到对应 case folder（可选）
    由于历史命名可能不统一，这里做“模糊匹配”：
    - folder name 同时包含 query 的 sha256 后缀/前缀 与 candidate 的 sha256 后缀/前缀
    若找不到返回 None。
    """
    if not VIZ_PAIRS_DIR.exists():
        return None
    q = qid.replace("sha256:", "")
    c = cid.replace("sha256:", "")
    q_key = q[:12]
    c_key = c[:12]
    # 先直接扫一层子目录
    for d in VIZ_PAIRS_DIR.glob("*"):
        if d.is_dir():
            name = d.name.lower()
            if (q_key.lower() in name) and (c_key.lower() in name):
                return d
    # 找不到则 None
    return None

def to_case_id(qid, cid):
    q = qid.replace("sha256:", "")[:12]
    c = cid.replace("sha256:", "")[:12]
    return f"q{q}__c{c}"

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_html(path: Path, title: str, payload: dict):
    # 简单 HTML：展示 JSON + 若有 viz 目录则链接提示（不强依赖浏览器）
    j = json.dumps(payload, ensure_ascii=False, indent=2)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:Consolas,Menlo,monospace; padding:16px;}}
pre{{white-space:pre-wrap; word-break:break-word; background:#111; color:#eee; padding:12px; border-radius:8px;}}
h1{{font-size:18px;}}
</style></head>
<body>
<h1>{title}</h1>
<pre>{j}</pre>
</body></html>
"""
    path.write_text(html, encoding="utf-8")

# -----------------------------
# Prepare output tree
# -----------------------------
ensure_clean_dir(OUT)
CASE_ROOT = OUT / "cases"
DISPUTE_DIR = CASE_ROOT / "dispute"
RANDOM_DIR  = CASE_ROOT / "random"
DISPUTE_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_DIR.mkdir(parents=True, exist_ok=True)
for v in VERS:
    (RANDOM_DIR / v).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load base inputs
# -----------------------------
assert PAIRS_FILE.exists(), f"Missing: {PAIRS_FILE}"
assert B_FILE.exists(), f"Missing: {B_FILE}"
for v in VERS:
    assert T_FILE(v).exists(), f"Missing: {T_FILE(v)}"
    assert C_FILE(v).exists(), f"Missing: {C_FILE(v)}"
    assert P_FILE(v).exists(), f"Missing: {P_FILE(v)}"

pairs = [r for r in iter_jsonl(PAIRS_FILE)]
pairs_map = {(r["query_image_id"], r["candidate_image_id"]): r for r in pairs}

A_map = load_map_pair(A_RANKED) if A_RANKED.exists() else load_map_pair(A_RAW)
B_map = load_map_pair(B_FILE)

T_maps = {v: load_map_pair(T_FILE(v)) for v in VERS}
C_maps = {v: load_map_pair(C_FILE(v)) for v in VERS}
P_maps = {v: load_map_pair(P_FILE(v)) for v in VERS}

# -----------------------------
# Build decision matrix to find disputes
# -----------------------------
keys = sorted(list(pairs_map.keys()))
dec_mat = {}
for k in keys:
    row = {}
    for v in VERS:
        t = T_maps[v].get(k)
        row[v] = safe_get(t, "final_decision", default=None)
    dec_mat[k] = row

dispute_keys = [k for k in keys if len(set([dec_mat[k][v] for v in VERS])) > 1]
all_agree_keys = [k for k in keys if len(set([dec_mat[k][v] for v in VERS])) == 1]

rng = np.random.RandomState(SEED)

# sample disputes (if fewer than N_DISPUTE, take all)
if len(dispute_keys) > 0:
    sample_dispute = [dispute_keys[i] for i in rng.choice(len(dispute_keys), size=min(N_DISPUTE, len(dispute_keys)), replace=False)]
else:
    sample_dispute = []

# random per version: sample from all keys (not limited to disputes)
sample_random = {}
for v in VERS:
    sample_random[v] = [keys[i] for i in rng.choice(len(keys), size=min(N_RANDOM_PER_VER, len(keys)), replace=False)]

# -----------------------------
# Case pack builder
# -----------------------------
def build_case_payload(k):
    qid, cid = k
    pair = pairs_map.get(k, {})
    out = {
        "key": {
            "query_image_id": qid,
            "candidate_image_id": cid,
        },
        "paths": {
            "query_path": pair.get("query_path"),
            "candidate_path": pair.get("candidate_path"),
        },
        "pool_meta": {
            "source": pair.get("source"),
            "pool_rank": pair.get("pool_rank"),
        },
        "A_clip": None,
        "B_8b_review": None,
        "versions": {},
        "viz_hint": None,
    }

    # A (ranked if exists)
    a = A_map.get(k)
    if a:
        out["A_clip"] = {
            "schema": a.get("schema"),
            "clip_similarity": safe_get(a, "clip_similarity"),
            "clip_rank": safe_get(a, "clip_rank", "rank"),
            "A_decision": safe_get(a, "A_decision", "decision"),
        }

    # B (strict schema)
    b = B_map.get(k)
    if b:
        out["B_8b_review"] = {
            "schema": b.get("schema"),
            "criteria": b_dims_strict(b),
            "final_recommendation": b.get("final_recommendation"),
            "confidence": b.get("confidence"),
            "final_reason": b.get("final_reason"),
        }

    # per-version: C/P + Teacher
    for v in VERS:
        c = C_maps[v].get(k)
        p = P_maps[v].get(k)
        t = T_maps[v].get(k)
        out["versions"][v] = {
            "C_maskclip": {
                "schema": c.get("schema") if c else None,
                "mask_clip_raw": safe_get(c, "mask_clip_raw", "clip_raw", "raw"),
                "mask_clip_masked": safe_get(c, "mask_clip_masked", "clip_masked", "masked"),
                "mask_clip_delta": safe_get(c, "mask_clip_delta", "delta", "mask_delta", "delta_sim"),
                "C_decision": safe_get(c, "C_decision", "decision", "final_decision"),
                "C_explain": safe_get(c, "C_explain", "explain_zh", "reason_zh", "explain"),
            },
            "P_polygon": {
                "schema": p.get("schema") if p else None,
                "noise_detected": safe_get(p, "noise_detected", "noise_decision"),
                "cover_frac_raw": safe_get(p, "cover_frac_raw"),
                "cover_frac_capped": safe_get(p, "cover_frac_capped"),
                "polygon_count": len(safe_get(p, "polygons", "polygon_list", default=[])) if p else None,
            },
            "Teacher_30b": {
                "schema": t.get("schema") if t else None,
                "final_decision": safe_get(t, "final_decision"),
                "confidence": safe_get(t, "confidence"),
                "adopted_from": safe_get(t, "adopted_from"),
                "overruled": safe_get(t, "overruled"),
                "reason": safe_get(t, "reason"),
            }
        }

    # optional viz hint
    viz_dir = find_viz_dir_for_pair(qid, cid)
    if viz_dir:
        out["viz_hint"] = str(viz_dir)

    return out

# -----------------------------
# Write dispute cases
# -----------------------------
dispute_manifest = []
for k in sample_dispute:
    qid, cid = k
    cid_str = to_case_id(qid, cid)
    case_dir = DISPUTE_DIR / cid_str
    case_dir.mkdir(parents=True, exist_ok=True)

    payload = build_case_payload(k)

    # small derived fields for quick scan
    decs = {v: payload["versions"][v]["Teacher_30b"]["final_decision"] for v in VERS}
    payload["_quick"] = {
        "teacher_decisions": decs,
        "n_unique_teacher_decisions": len(set(decs.values())),
    }

    write_json(case_dir / "case.json", payload)
    write_html(case_dir / "case.html", f"DISPUTE CASE {cid_str}", payload)

    dispute_manifest.append({
        "case_id": cid_str,
        "query_image_id": qid,
        "candidate_image_id": cid,
        **{f"{v}_decision": decs[v] for v in VERS},
        "viz_hint": payload.get("viz_hint"),
        "case_dir": str(case_dir),
    })

df_dispute_manifest = pd.DataFrame(dispute_manifest)
df_dispute_manifest.to_csv(OUT / "manifest_dispute.csv", index=False, encoding="utf-8-sig")
print("[DONE] manifest_dispute.csv ->", OUT / "manifest_dispute.csv")

# -----------------------------
# Write random cases per version
# -----------------------------
random_manifest = []
for v in VERS:
    for k in sample_random[v]:
        qid, cid = k
        cid_str = to_case_id(qid, cid)
        case_dir = (RANDOM_DIR / v / cid_str)
        case_dir.mkdir(parents=True, exist_ok=True)

        payload = build_case_payload(k)

        # quick focus on that version
        payload["_quick"] = {
            "focus_version": v,
            "teacher_decision": payload["versions"][v]["Teacher_30b"]["final_decision"],
            "teacher_confidence": payload["versions"][v]["Teacher_30b"]["confidence"],
            "teacher_adopted_from": payload["versions"][v]["Teacher_30b"]["adopted_from"],
            "teacher_reason": payload["versions"][v]["Teacher_30b"]["reason"],
        }

        write_json(case_dir / "case.json", payload)
        write_html(case_dir / "case.html", f"RANDOM CASE [{v}] {cid_str}", payload)

        random_manifest.append({
            "version": v,
            "case_id": cid_str,
            "query_image_id": qid,
            "candidate_image_id": cid,
            "teacher_decision": payload["versions"][v]["Teacher_30b"]["final_decision"],
            "teacher_confidence": payload["versions"][v]["Teacher_30b"]["confidence"],
            "viz_hint": payload.get("viz_hint"),
            "case_dir": str(case_dir),
        })

df_random_manifest = pd.DataFrame(random_manifest)
df_random_manifest.to_csv(OUT / "manifest_random.csv", index=False, encoding="utf-8-sig")
print("[DONE] manifest_random.csv ->", OUT / "manifest_random.csv")

# -----------------------------
# Summary stats (for audit)
# -----------------------------
# overall disputes
all_dispute_rate = len(dispute_keys) / max(1, len(keys))
# per-version decision distribution
dist = {}
for v in VERS:
    cnt = Counter([safe_get(T_maps[v].get(k), "final_decision", default=None) for k in keys])
    total = sum(cnt.values())
    dist[v] = {k: int(cnt.get(k,0)) for k in ["ACCEPT","REJECT","AMBIGUOUS",None]}
    dist[v]["total"] = int(total)

summary = {
    "out_dir": str(OUT),
    "n_total_pairs": int(len(keys)),
    "n_dispute_pairs": int(len(dispute_keys)),
    "dispute_rate": float(all_dispute_rate),
    "sampled_dispute": int(len(sample_dispute)),
    "sampled_random_per_version": int(min(N_RANDOM_PER_VER, len(keys))),
    "teacher_decision_dist_counts": dist,
    "inputs": {
        "pairs_file": str(PAIRS_FILE),
        "A_file_used": str(A_RANKED if A_RANKED.exists() else A_RAW),
        "B_file": str(B_FILE),
        "teacher_files": {v: str(T_FILE(v)) for v in VERS},
        "C_files": {v: str(C_FILE(v)) for v in VERS},
        "P_files": {v: str(P_FILE(v)) for v in VERS},
        "viz_pairs_dir": str(VIZ_PAIRS_DIR) if VIZ_PAIRS_DIR.exists() else None,
    }
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("[DONE] summary.json ->", OUT / "summary.json")

print("\n===== CASE PACK DONE =====")
print("OUT:", OUT)
print("Dispute total:", len(dispute_keys), "| sampled:", len(sample_dispute))
print("Random per version:", min(N_RANDOM_PER_VER, len(keys)))

# %% [cell 2]
import json
import shutil
from pathlib import Path
import pandas as pd

# ============================
# Paths
# ============================
ROOT = Path(r"<PROTOTYPE303_ROOT>\V30Bjg\30Bfenqiziji")
MANI = ROOT / "manifest_dispute.csv"
CASES = ROOT / "cases" / "dispute"

OUT_CSV  = ROOT / "case_study_candidates.csv"
OUT_HTML = ROOT / "case_study_candidates_top20.html"
SEL_DIR  = ROOT / "case_study_selected"
SEL_DIR.mkdir(parents=True, exist_ok=True)

VERS = ["v1", "v2", "v3", "v3_1", "v4"]

assert MANI.exists(), f"Missing: {MANI}"
assert CASES.exists(), f"Missing: {CASES}"

df = pd.read_csv(MANI)

# ============================
# Load per-case JSON for richer signals
# ============================
def load_case(case_dir: Path):
    p = case_dir / "case.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def has_C_adopted(case_payload, v="v4"):
    try:
        ad = case_payload["versions"][v]["Teacher_30b"]["adopted_from"]
        if isinstance(ad, list):
            return "C" in ad
        if isinstance(ad, str):
            return "C" in ad
    except Exception:
        pass
    return False

def get_cover(case_payload, v="v4"):
    # cover_frac_raw / capped 可能不存在于某些版本或字段名变体，按冻结 schema 取
    try:
        p = case_payload["versions"][v]["P_polygon"]
        raw = p.get("cover_frac_raw", None)
        cap = p.get("cover_frac_capped", None)
        return raw, cap
    except Exception:
        return None, None

def get_delta(case_payload, v="v4"):
    try:
        c = case_payload["versions"][v]["C_maskclip"]
        return c.get("mask_clip_delta", None)
    except Exception:
        return None

# ============================
# Scoring: prioritize "v4 shows meaningful change + uses C evidence"
# ============================
rows = []
for _, r in df.iterrows():
    case_id = r["case_id"]
    case_dir = Path(r["case_dir"])
    payload = load_case(case_dir)
    if payload is None:
        continue

    # teacher decisions
    d = {v: r.get(f"{v}_decision") for v in VERS}
    # focus patterns
    v4_dec = d["v4"]
    v1_dec = d["v1"]

    # adopted-from contains C (v4)
    c_adopt = has_C_adopted(payload, "v4")

    # cover and delta (v4)
    cover_raw, cover_cap = get_cover(payload, "v4")
    delta = get_delta(payload, "v4")

    # score rules (可解释：每条都是论文可写理由)
    score = 0
    tags = []

    # 1) v4 changes decision vs v1 (strong signal)
    if (v4_dec is not None) and (v1_dec is not None) and (v4_dec != v1_dec):
        score += 5
        tags.append("v4!=v1")

    # 2) v4 differs from majority (rare but interesting)
    decs = [d[v] for v in VERS if d[v] is not None]
    maj = max(set(decs), key=decs.count) if decs else None
    if maj and v4_dec and v4_dec != maj:
        score += 3
        tags.append("v4!=majority")

    # 3) v4 adopted C (noise-aware evidence used)
    if c_adopt:
        score += 4
        tags.append("v4_adopt_C")

    # 4) cover raw high or cap hit (indicates polygon/coverage meaningful)
    try:
        if cover_raw is not None and float(cover_raw) >= 0.5:
            score += 2
            tags.append("cover_raw>=0.5")
    except: pass

    try:
        if cover_cap is not None and float(cover_cap) >= 0.2:
            score += 2
            tags.append("cover_cap_hit")
    except: pass

    # 5) delta strongly negative (boundary sensitivity)
    try:
        if delta is not None and float(delta) <= -0.2:
            score += 2
            tags.append("delta<=-0.2")
    except: pass

    # keep some context
    qpath = payload.get("paths", {}).get("query_path")
    cpath = payload.get("paths", {}).get("candidate_path")
    v4_conf = payload["versions"]["v4"]["Teacher_30b"].get("confidence", None)
    v4_reason = payload["versions"]["v4"]["Teacher_30b"].get("reason", None)

    rows.append({
        "case_id": case_id,
        "score": score,
        "tags": "|".join(tags),
        "v1_dec": v1_dec,
        "v2_dec": d["v2"],
        "v3_dec": d["v3"],
        "v3_1_dec": d["v3_1"],
        "v4_dec": v4_dec,
        "v4_conf": v4_conf,
        "v4_adopt_C": c_adopt,
        "cover_raw_v4": cover_raw,
        "cover_cap_v4": cover_cap,
        "delta_v4": delta,
        "query_path": qpath,
        "candidate_path": cpath,
        "v4_reason": v4_reason,
        "case_dir": str(case_dir),
    })

cand = pd.DataFrame(rows).sort_values(["score", "v4_adopt_C"], ascending=[False, False])

# Save full candidate list
cand.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("[DONE] case_study_candidates.csv ->", OUT_CSV)

# ============================
# Export Top-20 HTML + copy selected cases
# ============================
TOPK = 20
top = cand.head(TOPK).copy()

# Copy files to a clean selected folder (overwrite)
if SEL_DIR.exists():
    # clear only inside SEL_DIR
    for x in SEL_DIR.glob("*"):
        if x.is_file():
            x.unlink()
        else:
            shutil.rmtree(x, ignore_errors=True)

for _, r in top.iterrows():
    src = Path(r["case_dir"])
    dst = SEL_DIR / r["case_id"]
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ["case.json", "case.html"]:
        sp = src / fn
        if sp.exists():
            shutil.copy2(sp, dst / fn)

# HTML table
def esc(s):
    if s is None: return ""
    s = str(s)
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

html_rows = []
for _, r in top.iterrows():
    html_rows.append(f"""
<tr>
  <td>{esc(r['case_id'])}</td>
  <td>{esc(r['score'])}</td>
  <td>{esc(r['tags'])}</td>
  <td>{esc(r['v1_dec'])}</td>
  <td>{esc(r['v4_dec'])}</td>
  <td>{esc(r['v4_conf'])}</td>
  <td>{esc(r['v4_adopt_C'])}</td>
  <td>{esc(r['cover_raw_v4'])}</td>
  <td>{esc(r['cover_cap_v4'])}</td>
  <td>{esc(r['delta_v4'])}</td>
  <td style="max-width:520px; white-space:pre-wrap;">{esc(r['v4_reason'])}</td>
  <td style="max-width:520px; white-space:pre-wrap;">{esc(r['query_path'])}<br>{esc(r['candidate_path'])}</td>
</tr>
""")

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>PRF 303 Case Study Candidates (Top {TOPK})</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f3f3f3; position: sticky; top: 0; }}
</style>
</head>
<body>
<h2>PRF 303 Case Study Candidates (Top {TOPK})</h2>
<p>Source: {esc(str(MANI))}</p>
<p>Selected cases copied to: {esc(str(SEL_DIR))}</p>
<table>
<thead>
<tr>
  <th>case_id</th><th>score</th><th>tags</th>
  <th>v1_dec</th><th>v4_dec</th><th>v4_conf</th>
  <th>v4_adopt_C</th><th>cover_raw_v4</th><th>cover_cap_v4</th><th>delta_v4</th>
  <th>v4_reason</th><th>paths</th>
</tr>
</thead>
<tbody>
{''.join(html_rows)}
</tbody>
</table>
</body></html>
"""
OUT_HTML.write_text(html, encoding="utf-8")
print("[DONE] case_study_candidates_top20.html ->", OUT_HTML)

print("\n===== DONE =====")
print("Top cases copied to:", SEL_DIR)

# %% [cell 3]

