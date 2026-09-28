# AUTO-EXPORTED FROM ORIGINAL JUPYTER NOTEBOOK
# This file is DERIVED and was not executed during export.
# Source notebook: PRF2.ipynb

# %% [cell 1]
# ===== Cell 0: Global paths & sanity =====
from pathlib import Path

# ---- roots ----
PRF_ROOT = Path(r"<PRF_ROOT>")
DATA_303 = PRF_ROOT / "archive" / "303"
TEST_ROOT = PRF_ROOT / "303test"

CKPT = TEST_ROOT / "checkpoints"
OUTS = TEST_ROOT / "outputs"
RUNS = TEST_ROOT / "runs"
VIZ  = TEST_ROOT / "viz"

for d in [CKPT, OUTS, RUNS, VIZ]:
    d.mkdir(parents=True, exist_ok=True)

print("[OK] PRF_ROOT :", PRF_ROOT)
print("[OK] DATA_303 :", DATA_303)
print("[OK] TEST_ROOT:", TEST_ROOT)

# ---- dataset sanity ----
for name in ["303w", "303u"]:
    p = DATA_303 / name
    assert p.exists(), f"Missing {p}"
    imgs = list(p.glob("*.jpg")) + list(p.glob("*.png"))
    print(f"[DATA] {name}: {len(imgs)} images")

print("[READY] environment looks sane")

# %% [cell 2]
# ===== Cell 1 (LOCAL-ONLY, resumable): CLIP embedding for 303u =====
import os, json, time, hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor

# --------- IMPORTANT: force local-only ---------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 可选：把 huggingface cache 强制指向你的 <PRF_CHECKPOINT_ROOT>\cache\hf（避免去 C:）
os.environ["HF_HOME"] = r"<PRF_CHECKPOINT_ROOT>\cache\hf"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[DEVICE]", DEVICE)

# 你本地已下载的 CLIP 目录（按你截图）
CLIP_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert CLIP_DIR.exists(), f"Missing local CLIP dir: {CLIP_DIR}"
assert (CLIP_DIR / "config.json").exists(), "Missing config.json"
assert (CLIP_DIR / "preprocessor_config.json").exists(), "Missing preprocessor_config.json"
# model 文件名可能是 model.safetensors 或 pytorch_model.bin，二者其一存在即可
assert (CLIP_DIR / "model.safetensors").exists() or (CLIP_DIR / "pytorch_model.bin").exists(), \
       "Missing model weights (model.safetensors or pytorch_model.bin)"

print("[OK] using local CLIP:", CLIP_DIR)

# --------- load local model/processor (NO INTERNET) ---------
model = CLIPModel.from_pretrained(str(CLIP_DIR), local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
model.eval()
print("[OK] model/processor loaded")

# --------- paths ---------
OUT_FILE = CKPT / "clip_emb_303u.jsonl"
ERR_FILE = CKPT / "clip_emb_303u_errors.jsonl"

# resume keys
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["path"])
            except:
                pass
print("[INFO] already done:", len(done))

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()

imgs = list((DATA_303 / "303u").glob("*.jpg")) + list((DATA_303 / "303u").glob("*.png"))
assert len(imgs) > 0, "No images found in 303u"
print("[INFO] total images:", len(imgs))

# warm-up (1 image) to avoid “black screen”
img0 = Image.open(imgs[0]).convert("RGB")
inputs0 = processor(images=img0, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _ = model.get_image_features(**inputs0)
print("[OK] warm-up done")

n_ok = 0
n_fail = 0
t0 = time.time()

with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
    for i, p in enumerate(tqdm(imgs, desc="CLIP emb 303u (local)", ncols=90)):
        sp = str(p)
        if sp in done:
            continue
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            fout.write(json.dumps({
                "schema": "clip_image_emb_v1",
                "image_id": sha256_of_file(p),
                "path": sp,
                "embedding": emb[0].detach().cpu().tolist()
            }) + "\n")
            fout.flush()
            done.add(sp)
            n_ok += 1

            # heartbeat
            if n_ok % 10 == 0:
                dt = time.time() - t0
                print(f"[HB] ok={n_ok} fail={n_fail} | elapsed={dt:.1f}s | last={p.name}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "clip_image_emb_v1_error",
                "path": sp,
                "error": str(e)
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] clip emb finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 3]
# ===== Cell 2 (resumable): CLIP Top-5 retrieval for 303w over 303u embeddings =====
import os, json, time, hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor

# --------- local-only (keep consistent) ---------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"<PRF_CHECKPOINT_ROOT>\cache\hf"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[DEVICE]", DEVICE)

CLIP_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert CLIP_DIR.exists(), f"Missing local CLIP dir: {CLIP_DIR}"
print("[OK] using local CLIP:", CLIP_DIR)

model = CLIPModel.from_pretrained(str(CLIP_DIR), local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
model.eval()
print("[OK] model/processor loaded")

# --------- inputs/outputs ---------
EMB_FILE = CKPT / "clip_emb_303u.jsonl"
OUT_FILE = CKPT / "clip_topk_303w.jsonl"
ERR_FILE = CKPT / "clip_topk_303w_errors.jsonl"

assert EMB_FILE.exists() and EMB_FILE.stat().st_size > 0, f"Empty emb file: {EMB_FILE}"

# --------- load 303u embeddings into a tensor ---------
print("[INFO] loading 303u embeddings ...")
u_ids = []
u_paths = []
u_embs = []

with EMB_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        u_ids.append(r["image_id"])
        u_paths.append(r["path"])
        u_embs.append(r["embedding"])

assert len(u_embs) > 0, "No embeddings loaded from clip_emb_303u.jsonl"
u_embs = torch.tensor(u_embs, dtype=torch.float32, device=DEVICE)  # (N, D)
print("[OK] loaded", len(u_embs), "embeddings, dim=", u_embs.shape[1])

# --------- resume done-set ---------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["query_path"])
            except:
                pass
print("[INFO] already done:", len(done))

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()

# --------- query list (303w) ---------
q_imgs = list((DATA_303 / "303w").glob("*.jpg")) + list((DATA_303 / "303w").glob("*.png"))
assert len(q_imgs) > 0, "No images found in 303w"
print("[INFO] total queries (303w):", len(q_imgs))

# warm-up
img0 = Image.open(q_imgs[0]).convert("RGB")
inputs0 = processor(images=img0, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _ = model.get_image_features(**inputs0)
print("[OK] warm-up done")

K = 5
n_ok = 0
n_fail = 0
t0 = time.time()

with OUT_FILE.open("a", encoding="utf-8") as fout, ERR_FILE.open("a", encoding="utf-8") as ferr:
    for i, q_path in enumerate(tqdm(q_imgs, desc="CLIP Top-5 for 303w", ncols=90)):
        qsp = str(q_path)
        if qsp in done:
            continue

        try:
            img = Image.open(q_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                q_emb = model.get_image_features(**inputs)
                q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)  # (1, D)

                # cosine similarity since both normalized: sim = dot
                sims = (u_embs @ q_emb[0])  # (N,)
                topv, topi = torch.topk(sims, k=K, largest=True)

            top = []
            for score, idx in zip(topv.detach().cpu().tolist(), topi.detach().cpu().tolist()):
                top.append({
                    "candidate_image_id": u_ids[idx],
                    "candidate_path": u_paths[idx],
                    "score": float(score),
                })

            out = {
                "schema": "clip_topk_v1",
                "query_image_id": sha256_of_file(q_path),
                "query_path": qsp,
                "k": K,
                "topk": top
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(qsp)
            n_ok += 1

            if n_ok % 10 == 0:
                dt = time.time() - t0
                print(f"[HB] ok={n_ok} fail={n_fail} | elapsed={dt:.1f}s | last={q_path.name}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "clip_topk_v1_error",
                "query_path": qsp,
                "error": str(e)
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] clip topk finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 4]
# ===== Cell 3 (resumable): 8B Suggest Top-5 candidates for each 303w query =====
import json, time, base64, hashlib
from pathlib import Path
from tqdm import tqdm
import requests

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ---------- inputs ----------
CLIP_TOPK_FILE = CKPT / "clip_topk_303w.jsonl"  # 用来提供候选池（更稳、更快）
assert CLIP_TOPK_FILE.exists() and CLIP_TOPK_FILE.stat().st_size > 0, f"Empty: {CLIP_TOPK_FILE}"

# ---------- outputs ----------
OUT_FILE = CKPT / "8b_suggest_topk_303w.jsonl"
ERR_FILE = CKPT / "8b_suggest_topk_303w_errors.jsonl"

# ---------- resume ----------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["query_path"])
            except:
                pass
print("[INFO] already done:", len(done))

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

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()

SYSTEM_PROMPT = "你是图像检索候选生成专家。严格只输出 JSON，不要输出任何解释性文字。"
USER_PROMPT_TMPL = """
给定一张查询图像（query）和一个候选池（candidate pool，含路径与CLIP相似度分数），
请你从候选池里选择最可能匹配 query 的 Top-5 候选，并输出严格 JSON。

输出 JSON schema（必须完全符合）：
{{
  "schema": "8b_suggest_topk_v1",
  "topk": [
    {{"rank": 1, "candidate_path": "...", "reason": "中文一句理由"}},
    ...
    {{"rank": 5, "candidate_path": "...", "reason": "中文一句理由"}}
  ]
}}

硬约束：
1) topk 必须恰好 5 条；rank 为 1..5 不重复；
2) candidate_path 必须来自候选池中的路径（不得发明新路径）；
3) reason 必须是中文一句话；
4) 只输出 JSON（不要 markdown，不要 ```）。
候选池如下（JSON 数组）：
{cand_pool_json}
""".strip()

def extract_json(txt: str) -> dict:
    txt = txt.strip()
    # 兼容万一模型吐了 ```json
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    return json.loads(txt)

def schema_check(res: dict, pool_paths: set):
    assert isinstance(res, dict)
    assert res.get("schema") == "8b_suggest_topk_v1"
    topk = res.get("topk")
    assert isinstance(topk, list) and len(topk) == 5
    ranks = [x.get("rank") for x in topk]
    assert sorted(ranks) == [1,2,3,4,5]
    for x in topk:
        p = x.get("candidate_path")
        assert isinstance(p, str) and p in pool_paths
        r = x.get("reason")
        assert isinstance(r, str) and len(r.strip()) > 0

def call_8b(query_path: Path, cand_pool: list) -> dict:
    cand_pool_json = json.dumps(cand_pool, ensure_ascii=False)
    user_prompt = USER_PROMPT_TMPL.format(cand_pool_json=cand_pool_json)

    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(query_path)}},
            ]},
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

# ---------- main ----------
n_ok = 0
n_fail = 0
t0 = time.time()

with CLIP_TOPK_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="8B suggest Top-5 (from CLIP pool)", ncols=90):
        row = json.loads(line)
        q_path = Path(row["query_path"])
        qsp = str(q_path)

        if qsp in done:
            continue

        try:
            assert q_path.exists(), f"Missing query image: {q_path}"

            # candidate pool from CLIP topk
            pool = []
            pool_paths = set()
            for item in row["topk"]:
                cp = item["candidate_path"]
                pool_paths.add(cp)
                pool.append({
                    "candidate_path": cp,
                    "clip_score": float(item["score"]),
                })

            res = call_8b(q_path, pool)
            schema_check(res, pool_paths)

            out = {
                "schema": "8b_suggest_topk_v1",
                "query_image_id": sha256_of_file(q_path),
                "query_path": qsp,
                "pool_size": len(pool),
                "pool": pool,
                "topk": res["topk"],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(qsp)
            n_ok += 1

            if n_ok % 5 == 0:
                dt = time.time() - t0
                print(f"[HB] ok={n_ok} fail={n_fail} | elapsed={dt:.1f}s | last={q_path.name}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "8b_suggest_topk_v1_error",
                "query_path": qsp,
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] 8B suggest finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 5]
# ===== Cell 4 (resumable): Union(CLIP-5 ∪ 8B-Suggest-5) -> union_topk + pairs_v2 =====
import json
from pathlib import Path
from tqdm import tqdm

CLIP_TOPK_FILE = CKPT / "clip_topk_303w.jsonl"
SUGG_TOPK_FILE = CKPT / "8b_suggest_topk_303w.jsonl"

UNION_FILE = CKPT / "union_topk_303w.jsonl"
PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
ERR_FILE   = CKPT / "cell4_union_errors.jsonl"

# ---- fail-fast ----
for f in [CLIP_TOPK_FILE, SUGG_TOPK_FILE]:
    assert f.exists() and f.stat().st_size > 0, f"Empty or missing input: {f}"

# ---- load CLIP topk into dict by query_path ----
clip_by_qpath = {}
with CLIP_TOPK_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qpath = r["query_path"]
        clip_by_qpath[qpath] = r

assert len(clip_by_qpath) > 0, "No records loaded from clip_topk_303w.jsonl"
print("[OK] loaded clip_topk queries:", len(clip_by_qpath))

# ---- resume (by query_path) ----
done_union = set()
done_pairs = set()

if UNION_FILE.exists():
    with UNION_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done_union.add(r["query_path"])
            except:
                pass

if PAIRS_FILE.exists():
    with PAIRS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done_pairs.add((r["query_image_id"], r["candidate_image_id"]))
            except:
                pass

print("[INFO] already done union:", len(done_union))
print("[INFO] already done pairs:", len(done_pairs))

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

# ---- main: iterate over 8B suggest (303 lines) and union with clip ----
n_union_ok = 0
n_union_fail = 0
n_pairs_new = 0

with SUGG_TOPK_FILE.open("r", encoding="utf-8") as fin, \
     UNION_FILE.open("a", encoding="utf-8") as funion, \
     PAIRS_FILE.open("a", encoding="utf-8") as fpairs, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="Cell4 Union: CLIP-5 ∪ 8B-5", ncols=90):
        s = json.loads(line)
        qpath = s["query_path"]

        if qpath in done_union:
            continue

        try:
            assert qpath in clip_by_qpath, f"Missing clip_topk entry for query_path={qpath}"
            c = clip_by_qpath[qpath]

            qid = c["query_image_id"]  # use CLIP's id (sha256)
            # sanity: use same query_path
            assert c["query_path"] == qpath, "Query path mismatch between clip and 8b"

            # build mapping: candidate_path -> candidate_image_id from CLIP topk pool
            path2id = {}
            clip_top = []
            for item in c["topk"]:
                path2id[item["candidate_path"]] = item["candidate_image_id"]
                clip_top.append({
                    "candidate_image_id": item["candidate_image_id"],
                    "candidate_path": item["candidate_path"],
                    "clip_score": float(item["score"]),
                    "source": ["clip"],
                })

            # 8B suggestions are restricted to the same pool (by our schema_check earlier)
            sugg_top = []
            for item in s["topk"]:
                cp = item["candidate_path"]
                cid = path2id.get(cp)
                assert cid is not None, f"8B suggested path not in clip pool: {cp}"
                sugg_top.append({
                    "candidate_image_id": cid,
                    "candidate_path": cp,
                    "rank": safe_int(item.get("rank")),
                    "reason": item.get("reason", ""),
                    "source": ["8b"],
                })

            # union: keep CLIP order first, then add 8B in rank order if new
            seen = set()
            union = []

            # CLIP first (sorted by clip_score desc already from topk)
            for it in clip_top:
                key = it["candidate_image_id"]
                if key in seen:
                    continue
                seen.add(key)
                union.append(it)

            # 8B next (rank 1..5)
            sugg_top = sorted(sugg_top, key=lambda x: x["rank"])
            for it in sugg_top:
                key = it["candidate_image_id"]
                if key in seen:
                    # merge source tag (clip + 8b)
                    for u in union:
                        if u["candidate_image_id"] == key:
                            if "8b" not in u.get("source", []):
                                u["source"] = u.get("source", []) + ["8b"]
                            # attach reason if not present
                            if "reason" not in u and it.get("reason"):
                                u["reason"] = it["reason"]
                            break
                    continue
                seen.add(key)
                union.append({
                    "candidate_image_id": it["candidate_image_id"],
                    "candidate_path": it["candidate_path"],
                    "rank": it["rank"],
                    "reason": it.get("reason", ""),
                    "source": ["8b"],
                })

            # cap at 10 unique (CLIP-5 ∪ 8B-5)
            union = union[:10]

            out_union = {
                "schema": "union_topk_v1",
                "query_image_id": qid,
                "query_path": qpath,
                "k_clip": 5,
                "k_8b": 5,
                "k_union": len(union),
                "union": union
            }

            funion.write(json.dumps(out_union, ensure_ascii=False) + "\n")
            funion.flush()
            done_union.add(qpath)
            n_union_ok += 1

            # expand into pairs_v2 (one line per candidate)
            for it in union:
                cid = it["candidate_image_id"]
                key = (qid, cid)
                if key in done_pairs:
                    continue

                q_p = Path(qpath)
                c_p = Path(it["candidate_path"])
                # hard existence check (fail-fast)
                assert q_p.exists(), f"query_path missing: {q_p}"
                assert c_p.exists(), f"candidate_path missing: {c_p}"

                pair = {
                    "schema": "pair_v2",
                    "query_image_id": qid,
                    "candidate_image_id": cid,
                    "query_path": str(q_p),
                    "candidate_path": str(c_p),
                    "source": it.get("source", []),
                }
                fpairs.write(json.dumps(pair, ensure_ascii=False) + "\n")
                fpairs.flush()
                done_pairs.add(key)
                n_pairs_new += 1

            if n_union_ok % 10 == 0:
                print(f"[HB] union_ok={n_union_ok} union_fail={n_union_fail} pairs_new={n_pairs_new} | last={Path(qpath).name}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "cell4_union_error",
                "query_path": qpath,
                "error": str(e)
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_union_fail += 1

print("[DONE] Cell4 union + pairs_v2 finished")
print("[STATS] union_ok:", n_union_ok, "union_fail:", n_union_fail, "pairs_new:", n_pairs_new)
print("[OUT UNION]", UNION_FILE, "bytes=", UNION_FILE.stat().st_size if UNION_FILE.exists() else 0)
print("[OUT PAIRS]", PAIRS_FILE, "bytes=", PAIRS_FILE.stat().st_size if PAIRS_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 6]
# ===== Cell 5 (FINAL): B1 8B Pair Review v3 (0-5 + reason + final_reason) =====
import json
import base64
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w_v2.jsonl"
OUT_FILE = CKPT / "b1_pair_review_303w_v3.jsonl"
ERR_FILE = CKPT / "b1_pair_review_303w_v3_errors.jsonl"

assert IN_FILE.exists() and IN_FILE.stat().st_size > 0, f"Empty input file: {IN_FILE}"

# ---------------- resume ----------------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except Exception:
                pass
print("[INFO] already done:", len(done))

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = "你是图像配对审核专家。严格只输出 JSON，不要输出任何解释性文字或 Markdown。"

# ---- 0~5 anchors (soft, but explicit) ----
ANCHORS = """
评分锚点（0-5，整数）：
0=完全不符合；1=极弱/几乎不支持；2=较弱/证据不足；3=中等/可接受但有明显疑点；4=较强/大体一致；5=极强/几乎确定。
""".strip()

USER_PROMPT = f"""
请对两张图像做配对审核，并只输出严格 JSON（不要 Markdown，不要 ```）。

输出 JSON schema（必须完全符合）：
{{
  "schema": "pair_review_v3",
  "criteria": {{
    "pixel_consistency": {{"score": 0, "reason": "中文一句话"}},
    "same_base_image": {{"score": 0, "reason": "中文一句话"}},
    "same_instance": {{"score": 0, "reason": "中文一句话"}},
    "strong_structure": {{"score": 0, "reason": "中文一句话"}},
    "weak_structure": {{"score": 0, "reason": "中文一句话"}}
  }},
  "final_recommendation": "ACCEPT|REJECT|AMBIGUOUS",
  "confidence": 0.0,
  "final_reason": "中文一句话最终理由"
}}

要求：
1) 每个维度必须给 score(0-5整数) + reason(中文一句话)；
2) final_reason 必须是中文一句话总结（不要简单重复某一个维度）；
3) confidence 为 0~1；
4) 只输出 JSON。
{ANCHORS}
""".strip()

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
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    return json.loads(txt)

def schema_check(res: dict):
    assert res.get("schema") == "pair_review_v3"
    assert "criteria" in res and isinstance(res["criteria"], dict)

    for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        assert k in res["criteria"], f"missing criteria.{k}"
        item = res["criteria"][k]
        assert isinstance(item, dict), f"criteria.{k} not dict"
        sc = int(item.get("score"))
        assert 0 <= sc <= 5, f"criteria.{k}.score out of range: {sc}"
        rs = item.get("reason")
        assert isinstance(rs, str) and len(rs.strip()) > 0, f"criteria.{k}.reason empty"

    assert res.get("final_recommendation") in ["ACCEPT","REJECT","AMBIGUOUS"]
    conf = float(res.get("confidence"))
    assert 0.0 <= conf <= 1.0
    fr = res.get("final_reason")
    assert isinstance(fr, str) and len(fr.strip()) > 0, "final_reason empty"

def call_8b(q_path: Path, c_path: Path) -> dict:
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": to_data_url(q_path)}},
                {"type": "image_url", "image_url": {"url": to_data_url(c_path)}},
            ]},
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

n_ok = 0
n_fail = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="B1 8B Pair Review v3 (0-5)", ncols=90):
        row = json.loads(line)
        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)
        if key in done:
            continue

        q_path = Path(row["query_path"])
        c_path = Path(row["candidate_path"])

        try:
            assert q_path.exists(), f"query_path missing: {q_path}"
            assert c_path.exists(), f"candidate_path missing: {c_path}"

            res = call_8b(q_path, c_path)
            schema_check(res)

            out = {
                "schema": "b1_pair_review_v3",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "source": row.get("source", []),
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": float(res["confidence"]),
                "final_reason": res["final_reason"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

            if n_ok % 10 == 0:
                print(f"[HB] ok={n_ok} fail={n_fail} | last={q_path.name}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "b1_pair_review_v3_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] B1 finished")
print("[STATS] success:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 7]
# ===== Cell 6: C-Evidence (ALL pairs) - polygon generation, LabelMe style =====
import json
import base64
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w_v2.jsonl"
OUT_FILE = CKPT / "b2_polygon_allpairs_v1.jsonl"
ERR_FILE = CKPT / "b2_polygon_allpairs_v1_errors.jsonl"

assert IN_FILE.exists() and IN_FILE.stat().st_size > 0, f"Empty input: {IN_FILE}"

# -------- resume --------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except:
                pass
print("[INFO] already done:", len(done))

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "你是图像证据分析专家，负责定位图像中的噪声或干扰区域。"
    "严格只输出 JSON，不要输出任何解释性文字或 Markdown。"
)

USER_PROMPT = """
给定一对图像（query 与 candidate），请你在【candidate 图像】中定位
可能影响相似性判断的噪声/干扰区域（如水印、文本覆盖、边框、遮挡、压缩伪影等），
并输出 LabelMe 风格的 polygon。

输出严格 JSON，schema 必须完全符合：

{
  "schema": "b2_polygon_v1",
  "polygon": {
    "points": [[x1, y1], [x2, y2], ...],
    "coord_type": "relative"
  },
  "reason": "中文一句话，说明该区域为何被认为是噪声/干扰",
  "confidence": 0.0
}

要求：
1) polygon.points 至少 3 个点，构成闭合多边形（无需重复首点）；
2) 坐标使用相对坐标（0~1），以左上角为 (0,0)，右下角为 (1,1)；
3) 如果未发现明显噪声，也必须给出一个【极小区域】polygon（例如角落很小一块），并在 reason 中说明“未发现显著噪声”；
4) confidence ∈ [0,1]；
5) 只输出 JSON。
""".strip()

def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suf == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    return json.loads(txt)

def schema_check(res: dict):
    assert res.get("schema") == "b2_polygon_v1"
    poly = res.get("polygon", {})
    pts = poly.get("points")
    assert isinstance(pts, list) and len(pts) >= 3
    for x, y in pts:
        assert 0.0 <= float(x) <= 1.0
        assert 0.0 <= float(y) <= 1.0
    assert poly.get("coord_type") == "relative"
    assert isinstance(res.get("reason"), str) and len(res["reason"].strip()) > 0
    conf = float(res.get("confidence"))
    assert 0.0 <= conf <= 1.0

def call_8b_polygon(q_path: Path, c_path: Path) -> dict:
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": to_data_url(q_path)}},
                {"type": "image_url", "image_url": {"url": to_data_url(c_path)}},
            ]},
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

# -------- main --------
n_ok = 0
n_fail = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="C-Evidence polygon (ALL pairs)", ncols=90):
        row = json.loads(line)
        key = (row["query_image_id"], row["candidate_image_id"])
        if key in done:
            continue

        q_path = Path(row["query_path"])
        c_path = Path(row["candidate_path"])

        try:
            assert q_path.exists()
            assert c_path.exists()

            res = call_8b_polygon(q_path, c_path)
            schema_check(res)

            out = {
                "schema": "b2_polygon_allpairs_v1",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "polygon": res["polygon"],
                "reason": res["reason"],
                "confidence": float(res["confidence"]),
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

            if n_ok % 20 == 0:
                print(f"[HB] ok={n_ok} fail={n_fail}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "b2_polygon_allpairs_v1_error",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] C-Evidence polygon finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 8]
# ===== Cell 7: Render polygon overlays + masks (ALL pairs) =====
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
VIZ_ROOT = Path(r"<PROTOTYPE303_ROOT>\viz") / "pairs"
VIZ_ROOT.mkdir(parents=True, exist_ok=True)

IN_FILE  = CKPT / "b2_polygon_allpairs_v1.jsonl"
ERR_FILE = CKPT / "cell7_overlay_errors.jsonl"

assert IN_FILE.exists() and IN_FILE.stat().st_size > 0, f"Empty input: {IN_FILE}"

def short_id(s: str, n=12) -> str:
    # sha256:abcd... -> abcd...
    if s.startswith("sha256:"):
        s = s.split("sha256:", 1)[1]
    return s[:n]

def rel_to_abs(points_rel, w, h):
    pts = []
    for x, y in points_rel:
        xa = float(x) * w
        ya = float(y) * h
        # clamp
        xa = max(0.0, min(w - 1.0, xa))
        ya = max(0.0, min(h - 1.0, ya))
        pts.append((xa, ya))
    return pts

# ---- resume: if candidate_overlay exists, skip ----
def is_done(out_dir: Path) -> bool:
    return (out_dir / "candidate_overlay.png").exists() and (out_dir / "candidate_mask.png").exists() and (out_dir / "meta.json").exists()

n_ok = 0
n_fail = 0

with IN_FILE.open("r", encoding="utf-8") as fin, ERR_FILE.open("a", encoding="utf-8") as ferr:
    for line in tqdm(fin, desc="Render overlays + masks (ALL pairs)", ncols=90):
        r = json.loads(line)

        qid = r["query_image_id"]
        cid = r["candidate_image_id"]
        qpath = Path(r["query_path"])
        cpath = Path(r["candidate_path"])

        try:
            assert qpath.exists(), f"Missing query image: {qpath}"
            assert cpath.exists(), f"Missing candidate image: {cpath}"

            qsid = short_id(qid)
            csid = short_id(cid)
            out_dir = VIZ_ROOT / f"{qsid}__{csid}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if is_done(out_dir):
                continue

            # ---- load images ----
            qimg = Image.open(qpath).convert("RGB")
            cimg = Image.open(cpath).convert("RGB")

            # ---- polygon points (relative -> absolute in candidate space) ----
            poly = r["polygon"]
            assert poly.get("coord_type") == "relative"
            pts_rel = poly["points"]
            cw, ch = cimg.size
            pts_abs = rel_to_abs(pts_rel, cw, ch)

            # ---- draw candidate overlay ----
            c_overlay = cimg.copy()
            draw = ImageDraw.Draw(c_overlay, "RGBA")
            # semi-transparent fill + outline
            draw.polygon(pts_abs, fill=(255, 0, 0, 80), outline=(255, 0, 0, 200))

            # ---- draw query overlay (optional, here we always save for audit) ----
            # We reuse relative points on query as well for audit consistency (not semantic alignment).
            qw, qh = qimg.size
            q_pts_abs = rel_to_abs(pts_rel, qw, qh)
            q_overlay = qimg.copy()
            qdraw = ImageDraw.Draw(q_overlay, "RGBA")
            qdraw.polygon(q_pts_abs, fill=(255, 0, 0, 60), outline=(255, 0, 0, 180))

            # ---- binary mask in candidate space ----
            mask = Image.new("L", (cw, ch), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.polygon(pts_abs, fill=255)  # 255 inside polygon

            # ---- save ----
            (out_dir / "candidate_overlay.png").write_bytes(b"")  # create early to avoid partial states ambiguity
            c_overlay.save(out_dir / "candidate_overlay.png")

            (out_dir / "candidate_mask.png").write_bytes(b"")
            mask.save(out_dir / "candidate_mask.png")

            (out_dir / "query_overlay.png").write_bytes(b"")
            q_overlay.save(out_dir / "query_overlay.png")

            meta = {
                "schema": "overlay_meta_v1",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(qpath),
                "candidate_path": str(cpath),
                "candidate_size": {"w": cw, "h": ch},
                "query_size": {"w": qw, "h": qh},
                "polygon": {
                    "coord_type": "relative",
                    "points_rel": pts_rel,
                    "points_abs_candidate": [[float(x), float(y)] for x, y in pts_abs],
                },
                "reason": r.get("reason", ""),
                "confidence": r.get("confidence", None),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[HB] ok={n_ok} fail={n_fail}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "cell7_overlay_error",
                "query_image_id": r.get("query_image_id", ""),
                "candidate_image_id": r.get("candidate_image_id", ""),
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] Cell7 render finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[VIZ_ROOT]", VIZ_ROOT)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 9]
# ===== Cell 8: Mask-CLIP re-score + Chinese explanation (ALL pairs) =====
import os, json, time, hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor

# --------- local-only (keep consistent) ---------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"<PRF_CHECKPOINT_ROOT>\cache\hf"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[DEVICE]", DEVICE)

CLIP_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert CLIP_DIR.exists(), f"Missing local CLIP dir: {CLIP_DIR}"
print("[OK] using local CLIP:", CLIP_DIR)

model = CLIPModel.from_pretrained(str(CLIP_DIR), local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
model.eval()
print("[OK] model/processor loaded")

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
VIZ_ROOT = Path(r"<PROTOTYPE303_ROOT>\viz") / "pairs"

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
POLY_FILE  = CKPT / "b2_polygon_allpairs_v1.jsonl"

OUT_FILE = CKPT / "c_maskclip_303w_v1.jsonl"
ERR_FILE = CKPT / "c_maskclip_303w_v1_errors.jsonl"

for f in [PAIRS_FILE, POLY_FILE]:
    assert f.exists() and f.stat().st_size > 0, f"Empty or missing: {f}"

# ---- build polygon reason map by (qid,cid) ----
poly_reason = {}
with POLY_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["query_image_id"], r["candidate_image_id"])
        poly_reason[key] = {
            "reason": r.get("reason", ""),
            "confidence": r.get("confidence", None),
            "polygon": r.get("polygon", None),
        }
assert len(poly_reason) > 0, "No polygon records loaded"
print("[OK] loaded polygon records:", len(poly_reason))

# ---- resume by (qid,cid) ----
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except:
                pass
print("[INFO] already done:", len(done))

def short_id(s: str, n=12) -> str:
    if s.startswith("sha256:"):
        s = s.split("sha256:", 1)[1]
    return s[:n]

def apply_mask_to_candidate(cimg: Image.Image, mask: Image.Image, mode="neutral_gray") -> Image.Image:
    """
    mask: L mode 0/255, 255 indicates noise region to be removed.
    mode:
      - neutral_gray: replace masked pixels with 127 gray (audit-friendly)
      - black: set to 0
      - blur: not used here to keep deterministic
    """
    c = cimg.convert("RGB")
    m = mask.convert("L")
    if mode == "neutral_gray":
        neutral = Image.new("RGB", c.size, (127, 127, 127))
        # where mask==255 -> take neutral, else original
        return Image.composite(neutral, c, m)
    elif mode == "black":
        black = Image.new("RGB", c.size, (0, 0, 0))
        return Image.composite(black, c, m)
    else:
        neutral = Image.new("RGB", c.size, (127, 127, 127))
        return Image.composite(neutral, c, m)

@torch.no_grad()
def clip_emb(img: Image.Image) -> torch.Tensor:
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    e = model.get_image_features(**inputs)
    e = e / e.norm(dim=-1, keepdim=True)
    return e[0]  # (D,)

def explain_delta(delta: float, poly_r: str) -> str:
    # 固定模板，避免模型再黑箱：解释只基于 delta 与 polygon reason
    if delta >= 0.10:
        return f"去除候选图噪声区域后相似度显著上升（Δ={delta:.2f}），说明原差异主要受噪声/覆盖影响；噪声提示：{poly_r}"
    if delta >= 0.03:
        return f"去除噪声区域后相似度小幅上升（Δ={delta:.2f}），噪声对匹配有一定干扰；噪声提示：{poly_r}"
    if delta <= -0.05:
        return f"去除噪声区域后相似度下降（Δ={delta:.2f}），说明被遮挡部分可能包含有效匹配线索或mask偏离；噪声提示：{poly_r}"
    return f"去除噪声区域后相似度变化不大（Δ={delta:.2f}），该噪声对匹配影响有限；噪声提示：{poly_r}"

# warm-up
tmp_img = Image.open(next(iter((Path(r"<PRF_ROOT>\archive\303\303w")).glob("*.*")))).convert("RGB")
_ = clip_emb(tmp_img)
print("[OK] warm-up done")

n_ok = 0
n_fail = 0
t0 = time.time()

with PAIRS_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="C Mask-CLIP (ALL pairs)", ncols=90):
        row = json.loads(line)
        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)

        if key in done:
            continue

        q_path = Path(row["query_path"])
        c_path = Path(row["candidate_path"])

        try:
            assert q_path.exists(), f"missing query: {q_path}"
            assert c_path.exists(), f"missing cand: {c_path}"
            assert key in poly_reason, "missing polygon for this pair"

            qsid = short_id(qid)
            csid = short_id(cid)
            out_dir = VIZ_ROOT / f"{qsid}__{csid}"
            mask_path = out_dir / "candidate_mask.png"
            assert mask_path.exists(), f"missing mask png: {mask_path}"

            qimg = Image.open(q_path).convert("RGB")
            cimg = Image.open(c_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # raw
            eq = clip_emb(qimg)
            ec = clip_emb(cimg)
            clip_raw = float((eq * ec).sum().detach().cpu().item())

            # masked candidate
            c_masked = apply_mask_to_candidate(cimg, mask, mode="neutral_gray")
            ec_m = clip_emb(c_masked)
            clip_masked = float((eq * ec_m).sum().detach().cpu().item())

            delta = clip_masked - clip_raw
            poly_r = poly_reason[key].get("reason", "")
            zh = explain_delta(delta, poly_r)

            out = {
                "schema": "c_maskclip_v1",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "mask_path": str(mask_path),
                "clip_raw": clip_raw,
                "clip_masked": clip_masked,
                "delta": float(delta),
                "explain_zh": zh,
                "polygon_reason": poly_r,
                "polygon_confidence": poly_reason[key].get("confidence", None),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

            if n_ok % 50 == 0:
                dt = time.time() - t0
                print(f"[HB] ok={n_ok} fail={n_fail} | elapsed={dt:.1f}s")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "c_maskclip_v1_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] Cell8 mask-clip finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 10]
# ===== Cell 9 (v2 FIX): 30B Audit Teacher (ALL pairs) - token replace (no .format braces issue) =====
import json
import base64
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")

PAIRS_FILE = CKPT / "pairs_303w_v2.jsonl"
B1_FILE    = CKPT / "b1_pair_review_303w_v3.jsonl"
C_FILE     = CKPT / "c_maskclip_303w_v1.jsonl"
POLY_FILE  = CKPT / "b2_polygon_allpairs_v1.jsonl"

OUT_FILE = CKPT / "teacher_30b_audit_303w_v3_v2.jsonl"
ERR_FILE = CKPT / "teacher_30b_audit_303w_v3_v2_errors.jsonl"

for f in [PAIRS_FILE, C_FILE, POLY_FILE]:
    assert f.exists() and f.stat().st_size > 0, f"Empty or missing: {f}"
assert B1_FILE.exists() and B1_FILE.stat().st_size > 0, f"Missing/empty B1 file: {B1_FILE}"

# -------- load B1 by (qid,cid) --------
b1 = {}
with B1_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        b1[(r["query_image_id"], r["candidate_image_id"])] = r
print("[OK] loaded B1 records:", len(b1))

# -------- load C --------
cmap = {}
with C_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cmap[(r["query_image_id"], r["candidate_image_id"])] = r
print("[OK] loaded C mask-clip records:", len(cmap))

# -------- load polygon --------
pmap = {}
with POLY_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pmap[(r["query_image_id"], r["candidate_image_id"])] = r
print("[OK] loaded polygon records:", len(pmap))

# -------- resume --------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except:
                pass
print("[INFO] already done:", len(done))

TEACHER_URL = "http://127.0.0.1:8081/v1/chat/completions"

SYSTEM_PROMPT = (
    "你是严格的图像配对裁判（Teacher）。你将看到两张图像以及上游证据（B1评审、Mask-CLIP变化、噪声区域描述）。"
    "你必须输出严格 JSON，不要输出任何解释性文字或 Markdown。"
)

ANCHORS = "评分锚点（0-5整数）：0=完全不符合；1=极弱；2=较弱；3=中等；4=较强；5=极强。"

USER_PROMPT_TMPL = """
请基于两张图像（query 与 candidate）以及下方证据，给出最终裁判结论，并只输出严格 JSON。

输出 JSON schema（必须完全符合）：
{
  "schema": "pair_review_v3",
  "criteria": {
    "pixel_consistency": {"score": 0, "reason": "中文一句话"},
    "same_base_image": {"score": 0, "reason": "中文一句话"},
    "same_instance": {"score": 0, "reason": "中文一句话"},
    "strong_structure": {"score": 0, "reason": "中文一句话"},
    "weak_structure": {"score": 0, "reason": "中文一句话"}
  },
  "final_recommendation": "ACCEPT|REJECT|AMBIGUOUS",
  "confidence": 0.0,
  "final_reason": "中文一句话最终理由"
}

要求：
1) 每个维度必须给 score(0-5整数) + reason（中文一句话）；
2) final_reason 必须是中文一句话总结（不要简单重复某一个维度）；
3) confidence ∈ [0,1]；
4) 只输出 JSON。

【ANCHORS】
证据（可能为空/缺失）：
- B1评审摘要：【B1_SUMMARY】
- 噪声区域提示：【POLY_REASON】
- Mask-CLIP：raw=【CLIP_RAW】, masked=【CLIP_MASKED】, delta=【DELTA】
- C路解释：【C_EXPLAIN】
""".strip()

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
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    return json.loads(txt)

def schema_check(res: dict):
    assert res.get("schema") == "pair_review_v3"
    assert "criteria" in res and isinstance(res["criteria"], dict)
    for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        assert k in res["criteria"]
        item = res["criteria"][k]
        sc = int(item.get("score"))
        assert 0 <= sc <= 5
        rs = item.get("reason")
        assert isinstance(rs, str) and len(rs.strip()) > 0
    assert res.get("final_recommendation") in ["ACCEPT","REJECT","AMBIGUOUS"]
    conf = float(res.get("confidence"))
    assert 0.0 <= conf <= 1.0
    fr = res.get("final_reason")
    assert isinstance(fr, str) and len(fr.strip()) > 0

def call_30b(q_path: Path, c_path: Path, user_prompt: str) -> dict:
    payload = {
        "model": "qwen3-vl-30b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(q_path)}},
                {"type": "image_url", "image_url": {"url": to_data_url(c_path)}},
            ]},
        ],
        "temperature": 0.0,
    }
    r = requests.post(TEACHER_URL, json=payload, timeout=360)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

def b1_to_summary(b1rec: dict) -> str:
    if not b1rec:
        return "MISSING"
    fr = b1rec.get("final_recommendation", "NA")
    cf = b1rec.get("confidence", None)
    crit = b1rec.get("criteria", {})
    def s(k):
        try: return int(crit[k]["score"])
        except: return -1
    scores = {k: s(k) for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]}
    return f"final={fr}, conf={cf}, scores={scores}, final_reason={b1rec.get('final_reason','')[:30]}"

def compute_should_call_teacher(b1rec: dict, crec: dict) -> bool:
    if not b1rec:
        return True
    if b1rec.get("final_recommendation") == "AMBIGUOUS":
        return True
    try:
        if float(b1rec.get("confidence", 0)) < 0.60:
            return True
    except:
        return True
    if abs(float(crec.get("delta", 0.0))) >= 0.08:
        return True
    return False

n_ok = 0
n_fail = 0

with PAIRS_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="30B Audit Teacher (ALL pairs)", ncols=90):
        row = json.loads(line)
        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)

        if key in done:
            continue

        q_path = Path(row["query_path"])
        c_path = Path(row["candidate_path"])

        try:
            assert q_path.exists(), f"missing query: {q_path}"
            assert c_path.exists(), f"missing cand: {c_path}"
            assert key in cmap, "missing C record"
            assert key in pmap, "missing polygon record"

            b1rec = b1.get(key, None)
            crec = cmap[key]
            polyrec = pmap[key]

            b1_summary = b1_to_summary(b1rec)
            poly_reason = polyrec.get("reason", "")
            clip_raw = float(crec["clip_raw"])
            clip_masked = float(crec["clip_masked"])
            delta = float(crec["delta"])
            c_explain = crec.get("explain_zh", "")

            should_call = compute_should_call_teacher(b1rec, crec)

            user_prompt = (USER_PROMPT_TMPL
                .replace("【ANCHORS】", ANCHORS)
                .replace("【B1_SUMMARY】", b1_summary)
                .replace("【POLY_REASON】", poly_reason)
                .replace("【CLIP_RAW】", f"{clip_raw:.4f}")
                .replace("【CLIP_MASKED】", f"{clip_masked:.4f}")
                .replace("【DELTA】", f"{delta:.4f}")
                .replace("【C_EXPLAIN】", c_explain)
            )

            res = call_30b(q_path, c_path, user_prompt)
            schema_check(res)

            out = {
                "schema": "teacher_30b_audit_v3",
                "audit_forced": True,
                "should_call_teacher": bool(should_call),
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "b1_present": b1rec is not None,
                "b1_summary": b1_summary,
                "polygon_reason": poly_reason,
                "maskclip": {
                    "clip_raw": clip_raw,
                    "clip_masked": clip_masked,
                    "delta": delta,
                    "explain_zh": c_explain,
                },
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": float(res["confidence"]),
                "final_reason": res["final_reason"],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

            if n_ok % 10 == 0:
                print(f"[HB] ok={n_ok} fail={n_fail}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "teacher_30b_audit_v3_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            n_fail += 1

print("[DONE] Cell9 teacher audit finished")
print("[STATS] ok:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE, "bytes=", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
print("[ERR]", ERR_FILE, "bytes=", ERR_FILE.stat().st_size if ERR_FILE.exists() else 0)

# %% [cell 11]
# ===== Cell 10: 303 summary + export tables =====
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
OUTS = Path(r"<PROTOTYPE303_ROOT>\outputs")
OUTS.mkdir(parents=True, exist_ok=True)

B1_FILE = CKPT / "b1_pair_review_303w_v3.jsonl"
C_FILE  = CKPT / "c_maskclip_303w_v1.jsonl"
T_FILE  = CKPT / "teacher_30b_audit_303w_v3_v2.jsonl"

for f in [B1_FILE, C_FILE, T_FILE]:
    assert f.exists() and f.stat().st_size > 0, f"Missing/empty: {f}"

def load_map(path: Path, key_fields=("query_image_id","candidate_image_id")):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = tuple(r[x] for x in key_fields)
            m[k] = r
    return m

b1 = load_map(B1_FILE)
c  = load_map(C_FILE)
t  = load_map(T_FILE)

print("[OK] B1:", len(b1), "C:", len(c), "Teacher:", len(t))

keys = sorted(list(set(t.keys())))  # teacher is full 1490

def crit_scores(rec, prefix):
    # return dict of five scores
    out = {}
    crit = rec.get("criteria", {})
    for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        v = None
        try:
            v = int(crit[k]["score"])
        except:
            v = None
        out[f"{prefix}_{k}"] = v
    return out

rows = []
for k in keys:
    qid, cid = k
    tr = t[k]
    cr = c.get(k, {})
    br = b1.get(k, None)

    row = {
        "query_image_id": qid,
        "candidate_image_id": cid,
        "audit_forced": bool(tr.get("audit_forced", True)),
        "should_call_teacher": bool(tr.get("should_call_teacher", False)),

        "teacher_final": tr.get("final_recommendation"),
        "teacher_conf": tr.get("confidence"),
        "teacher_reason": tr.get("final_reason"),

        "b1_present": br is not None,
        "b1_final": br.get("final_recommendation") if br else None,
        "b1_conf": br.get("confidence") if br else None,
        "b1_reason": br.get("final_reason") if br else None,

        "clip_raw": cr.get("clip_raw", None),
        "clip_masked": cr.get("clip_masked", None),
        "delta": cr.get("delta", None),
        "c_explain_zh": cr.get("explain_zh", None),

        "polygon_reason": tr.get("polygon_reason", None),

        "query_path": tr.get("query_path"),
        "candidate_path": tr.get("candidate_path"),
    }

    row.update(crit_scores(tr, "teacher"))
    if br:
        row.update(crit_scores(br, "b1"))
    else:
        for kk in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
            row[f"b1_{kk}"] = None

    # agreement flag
    row["agree_b1_teacher"] = (row["b1_final"] == row["teacher_final"]) if br else None

    rows.append(row)

df = pd.DataFrame(rows)

# ---- core stats ----
teacher_dist = df["teacher_final"].value_counts(dropna=False).to_dict()
b1_dist = df["b1_final"].value_counts(dropna=False).to_dict()

agree_rate = df.loc[df["b1_present"] == True, "agree_b1_teacher"].mean()
should_call_rate = df["should_call_teacher"].mean()

delta = pd.to_numeric(df["delta"], errors="coerce")
delta_stats = {
    "count": int(delta.notna().sum()),
    "mean": float(delta.mean()),
    "std": float(delta.std(ddof=0)),
    "p05": float(delta.quantile(0.05)),
    "p25": float(delta.quantile(0.25)),
    "p50": float(delta.quantile(0.50)),
    "p75": float(delta.quantile(0.75)),
    "p95": float(delta.quantile(0.95)),
    "min": float(delta.min()),
    "max": float(delta.max()),
}

summary = {
    "n_pairs_teacher": int(len(df)),
    "n_pairs_b1_present": int(df["b1_present"].sum()),
    "teacher_final_dist": teacher_dist,
    "b1_final_dist": b1_dist,
    "b1_teacher_agreement_rate_on_b1_present": None if np.isnan(agree_rate) else float(agree_rate),
    "should_call_teacher_rate_normal_logic": float(should_call_rate),
    "maskclip_delta_stats": delta_stats,
    "paths": {
        "b1": str(B1_FILE),
        "c_maskclip": str(C_FILE),
        "teacher": str(T_FILE),
        "viz_root": str(Path(r"<PROTOTYPE303_ROOT>\viz\pairs")),
    },
}

# ---- export tables ----
pairs_csv = OUTS / "table_303_pairs.csv"
df.to_csv(pairs_csv, index=False, encoding="utf-8-sig")

# agreement table
agree_df = df[df["b1_present"] == True][
    ["query_image_id","candidate_image_id","b1_final","teacher_final","agree_b1_teacher","b1_conf","teacher_conf","delta","polygon_reason"]
].copy()
agree_csv = OUTS / "table_303_agreement.csv"
agree_df.to_csv(agree_csv, index=False, encoding="utf-8-sig")

# delta bins
delta_bins = pd.cut(delta, bins=[-1.0,-0.10,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.10,1.0])
delta_bin_counts = delta_bins.value_counts().sort_index()
delta_bin_df = delta_bin_counts.reset_index()
delta_bin_df.columns = ["delta_bin","count"]
delta_bins_csv = OUTS / "delta_maskclip_stats.csv"
delta_bin_df.to_csv(delta_bins_csv, index=False, encoding="utf-8-sig")

# sample disagreements (top 50 by abs(delta) among disagreements) for audit
dis = df[(df["b1_present"]==True) & (df["agree_b1_teacher"]==False)].copy()
dis["abs_delta"] = pd.to_numeric(dis["delta"], errors="coerce").abs()
dis = dis.sort_values(["abs_delta"], ascending=False).head(50)

sample_jsonl = OUTS / "sample_disagreements.jsonl"
with sample_jsonl.open("w", encoding="utf-8") as f:
    for _, r in dis.iterrows():
        f.write(json.dumps({
            "query_image_id": r["query_image_id"],
            "candidate_image_id": r["candidate_image_id"],
            "b1_final": r["b1_final"],
            "teacher_final": r["teacher_final"],
            "delta": r["delta"],
            "polygon_reason": r["polygon_reason"],
            "c_explain_zh": r["c_explain_zh"],
            "viz_folder_hint": f"{r['query_image_id'][:18]}__{r['candidate_image_id'][:18]}",
            "query_path": r["query_path"],
            "candidate_path": r["candidate_path"],
        }, ensure_ascii=False) + "\n")

# summary json
summary_path = OUTS / "summary_303.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("[DONE] Cell10 summary exported")
print("[OUT]", pairs_csv)
print("[OUT]", agree_csv)
print("[OUT]", delta_bins_csv)
print("[OUT]", sample_jsonl)
print("[OUT]", summary_path)
print("\n[KEY STATS]")
print(json.dumps({k: summary[k] for k in ["n_pairs_teacher","n_pairs_b1_present","b1_teacher_agreement_rate_on_b1_present","should_call_teacher_rate_normal_logic","teacher_final_dist","maskclip_delta_stats"]}, ensure_ascii=False, indent=2))

# %% [cell 12]
# ===== Cell 6-v2: polygon (type-aware + constrained) =====
import json, base64, requests, time
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
VIZ_V2 = Path(r"<PROTOTYPE303_ROOT>\viz_v2")
VIZ_V2.mkdir(parents=True, exist_ok=True)

PAIRS = CKPT / "pairs_303w_v2.jsonl"
OUT  = CKPT / "b2_polygon_allpairs_v2.jsonl"
ERR  = CKPT / "b2_polygon_allpairs_v2_errors.jsonl"

assert PAIRS.exists() and PAIRS.stat().st_size > 0

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"  # 8B 负责画证据（快），后续仍可用 teacher 审计
MAX_COVER_FRAC = 0.30  # 覆盖面积上限（可调 0.20~0.35）

SYSTEM = (
    "你是视觉审计专家。任务：仅标注应忽略的“噪声/覆盖/水印/边框”区域，用于证据审计与Mask-CLIP。"
    "必须输出严格JSON，不要Markdown。"
)

USER_TMPL = f"""
给你两张图：query 与 candidate。我们只对 candidate 标注“噪声区域”（watermark/overlay_text/border等）。
你必须先判断噪声类型 noise_type，然后再给出至多3个多边形（LabelMe风格 points）。

强约束（必须遵守）：
1) 只标注噪声/覆盖/水印/边框等应忽略区域；不要标主体结构（人/建筑/器物/主体文字内容除非它是水印覆盖）。
2) polygon 数量：0~3 个。
3) 每个 polygon 要尽量紧致（不要用大矩形糊弄），尽量贴合噪声形状。
4) 总覆盖面积（所有polygon并集）必须 <= {MAX_COVER_FRAC:.2f}（相对整图面积）。如果噪声是全图性质，请将 noise_type="global"，并返回空 polygons=[]。
5) 输出坐标归一化到 [0,1]：points = [[x,y], ...]，按顺时针，至少4个点，闭合不必重复首点。
6) 每个 polygon 给一句中文 reason_zh；并给全局 reason_zh（中文一句话总结为什么这些区域应忽略）。

noise_type 枚举：
- "none": 几乎无噪声/无覆盖（polygons=[]）
- "overlay_text": 大字覆盖/水印字样/斜体字覆盖但局部
- "watermark": 典型水印logo/半透明图案（局部）
- "border": 边框/黑边/拼接边缘
- "compression": 块状压缩/马赛克（局部）
- "global": 全图覆盖/全图强干扰（禁止画polygon）
- "unknown": 无法确定（保守返回少量局部polygon或为空）

输出严格 JSON：
{{
  "schema": "polygon_v2",
  "noise_type": "none|overlay_text|watermark|border|compression|global|unknown",
  "polygons": [
    {{
      "label": "noise",
      "points": [[0.1,0.2],[...]],
      "reason_zh": "中文一句话"
    }}
  ],
  "reason_zh": "中文一句话总结"
}}
""".strip()

def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    mime = "image/jpeg" if suf in [".jpg",".jpeg"] else "image/png" if suf==".png" else "image/webp" if suf==".webp" else "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower()=="json":
            txt="\n".join(lines[1:])
    return json.loads(txt)

def schema_check(d: dict):
    assert d.get("schema")=="polygon_v2"
    assert d.get("noise_type") in ["none","overlay_text","watermark","border","compression","global","unknown"]
    assert "polygons" in d and isinstance(d["polygons"], list)
    assert isinstance(d.get("reason_zh",""), str)
    if d["noise_type"]=="global":
        assert len(d["polygons"])==0
    assert len(d["polygons"])<=3
    for poly in d["polygons"]:
        assert poly.get("label")=="noise"
        pts = poly.get("points")
        assert isinstance(pts, list) and len(pts)>=4
        for x,y in pts:
            x=float(x); y=float(y)
            assert 0.0<=x<=1.0 and 0.0<=y<=1.0
        assert isinstance(poly.get("reason_zh",""), str)

def call_8b_polygon(q_path: Path, c_path: Path) -> dict:
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":USER_TMPL},
                {"type":"image_url","image_url":{"url":to_data_url(q_path)}},
                {"type":"image_url","image_url":{"url":to_data_url(c_path)}},
            ]}
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

# resume
done=set()
if OUT.exists():
    with OUT.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r=json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except: pass
print("[INFO] already done:", len(done))

ok=0; fail=0
with PAIRS.open("r", encoding="utf-8") as fin, \
     OUT.open("a", encoding="utf-8") as fout, \
     ERR.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="C-v2 polygon (constrained)", ncols=90):
        row=json.loads(line)
        qid=row["query_image_id"]; cid=row["candidate_image_id"]
        key=(qid,cid)
        if key in done: 
            continue

        q_path=Path(row["query_path"]); c_path=Path(row["candidate_path"])
        try:
            assert q_path.exists() and c_path.exists()
            res=call_8b_polygon(q_path,c_path)
            schema_check(res)

            out={
                "schema":"b2_polygon_allpairs_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "noise_type": res["noise_type"],
                "polygons": res["polygons"],
                "reason": res["reason_zh"],
                "max_cover_frac": MAX_COVER_FRAC,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n")
            fout.flush()
            done.add(key)
            ok += 1
            if ok % 50 == 0:
                print(f"[HB] ok={ok} fail={fail}")

        except Exception as e:
            ferr.write(json.dumps({
                "schema":"b2_polygon_allpairs_v2_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False)+"\n")
            ferr.flush()
            fail += 1

print("[DONE] C-v2 polygon finished")
print("[STATS] ok:", ok, "fail:", fail)
print("[OUT]", OUT, "bytes=", OUT.stat().st_size if OUT.exists() else 0)
print("[ERR]", ERR, "bytes=", ERR.stat().st_size if ERR.exists() else 0)

# %% [cell 13]
# ===== Cell 7-v2: render overlays to viz_v2 =====
import json
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
INP  = CKPT / "b2_polygon_allpairs_v2.jsonl"
ERR  = CKPT / "cell7_overlay_v2_errors.jsonl"
VIZ  = Path(r"<PROTOTYPE303_ROOT>\viz_v2\pairs")
VIZ.mkdir(parents=True, exist_ok=True)

assert INP.exists() and INP.stat().st_size > 0

def denorm_points(pts, w, h):
    return [(float(x)*w, float(y)*h) for x,y in pts]

ok=0; fail=0
with INP.open("r", encoding="utf-8") as fin, ERR.open("a", encoding="utf-8") as ferr:
    for line in tqdm(fin, desc="Render overlay v2", ncols=90):
        r=json.loads(line)
        qid=r["query_image_id"]; cid=r["candidate_image_id"]
        folder = VIZ / f"{qid[7:25]}__{cid[7:25]}"
        folder.mkdir(parents=True, exist_ok=True)

        try:
            c_path = Path(r["candidate_path"])
            img = Image.open(c_path).convert("RGBA")
            w,h = img.size

            overlay = Image.new("RGBA", (w,h), (0,0,0,0))
            dr = ImageDraw.Draw(overlay, "RGBA")

            for poly in r["polygons"]:
                pts = denorm_points(poly["points"], w, h)
                # 半透明红填充 + 红边
                dr.polygon(pts, fill=(255,0,0,80), outline=(255,0,0,200))

            out_img = Image.alpha_composite(img, overlay).convert("RGB")
            out_img.save(folder / "candidate_overlay.png", quality=95)

            # 也保存 meta
            meta = {
                "schema":"overlay_v2_meta",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "noise_type": r.get("noise_type"),
                "reason": r.get("reason"),
                "polygons": r.get("polygons"),
                "candidate_path": r.get("candidate_path"),
            }
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            ok += 1
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"overlay_v2_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False)+"\n")
            ferr.flush()
            fail += 1

print("[DONE] Cell7-v2 render finished")
print("[STATS] ok:", ok, "fail:", fail)
print("[VIZ_V2]", VIZ)
print("[ERR]", ERR, "bytes=", ERR.stat().st_size if ERR.exists() else 0)

# %% [cell 14]
# ===== Cell 8-v2: mask-clip using v2 polygons =====
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY = CKPT / "b2_polygon_allpairs_v2.jsonl"
OUT  = CKPT / "c_maskclip_303w_v2.jsonl"
ERR  = CKPT / "c_maskclip_303w_v2_errors.jsonl"

assert POLY.exists() and POLY.stat().st_size > 0

# ---- local CLIP ----
LOCAL_CLIP = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
model = CLIPModel.from_pretrained(str(LOCAL_CLIP)).eval()
processor = CLIPProcessor.from_pretrained(str(LOCAL_CLIP))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("[OK] CLIP loaded on", device)

def denorm_points(pts, w, h):
    return [(float(x)*w, float(y)*h) for x,y in pts]

def make_mask(img_size, polygons):
    w,h = img_size
    m = Image.new("L", (w,h), 0)
    dr = ImageDraw.Draw(m)
    for poly in polygons:
        pts = denorm_points(poly["points"], w, h)
        dr.polygon(pts, fill=255)
    return m

def apply_mask(img_rgb, mask_l):
    # hard mask: 将噪声区域涂黑（后续可升级 soft mask）
    arr = np.array(img_rgb).copy()
    m = np.array(mask_l) > 0
    arr[m] = 0
    return Image.fromarray(arr)

@torch.no_grad()
def clip_sim(img1, img2):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    sim = (feats[0] * feats[1]).sum().item()
    return sim

# resume
done=set()
if OUT.exists():
    with OUT.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r=json.loads(line)
                done.add((r["query_image_id"], r["candidate_image_id"]))
            except: pass
print("[INFO] already done:", len(done))

ok=0; fail=0
with POLY.open("r", encoding="utf-8") as fin, \
     OUT.open("a", encoding="utf-8") as fout, \
     ERR.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="Mask-CLIP v2", ncols=90):
        r=json.loads(line)
        qid=r["query_image_id"]; cid=r["candidate_image_id"]
        key=(qid,cid)
        if key in done:
            continue

        try:
            q = Image.open(r["query_path"]).convert("RGB")
            c = Image.open(r["candidate_path"]).convert("RGB")

            clip_raw = clip_sim(q, c)

            noise_type = r.get("noise_type")
            polys = r.get("polygons", [])

            if noise_type == "global" or len(polys)==0:
                clip_masked = clip_raw
                explain = f"noise_type={noise_type}, polygons=0 -> no masking"
            else:
                mask = make_mask(c.size, polys)
                c_masked = apply_mask(c, mask)
                clip_masked = clip_sim(q, c_masked)
                explain = f"noise_type={noise_type}, polygons={len(polys)}"

            out={
                "schema":"c_maskclip_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_raw": float(clip_raw),
                "clip_masked": float(clip_masked),
                "delta": float(clip_masked - clip_raw),
                "explain_zh": explain,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n")
            fout.flush()
            done.add(key)
            ok += 1

        except Exception as e:
            ferr.write(json.dumps({
                "schema":"c_maskclip_v2_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False)+"\n")
            ferr.flush()
            fail += 1

print("[DONE] Cell8-v2 mask-clip finished")
print("[STATS] ok:", ok, "fail:", fail)
print("[OUT]", OUT, "bytes=", OUT.stat().st_size if OUT.exists() else 0)
print("[ERR]", ERR, "bytes=", ERR.stat().st_size if ERR.exists() else 0)

# %% [cell 15]
import json
from pathlib import Path

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
PAIRS = CKPT/"pairs_303w_v2.jsonl"
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"
MC2 = CKPT/"c_maskclip_303w_v2.jsonl"

def keys(path):
    s=set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            s.add((r["query_image_id"], r["candidate_image_id"]))
    return s

k_pairs = keys(PAIRS)
k_poly2 = keys(POLY2)
k_mc2 = keys(MC2)

print("[COUNT] pairs:", len(k_pairs))
print("[COUNT] poly_v2:", len(k_poly2))
print("[COUNT] maskclip_v2:", len(k_mc2))

miss_poly = sorted(list(k_pairs - k_poly2))
miss_mc = sorted(list(k_pairs - k_mc2))

print("[MISSING] poly_v2 missing from pairs:", len(miss_poly))
print("[MISSING] maskclip_v2 missing from pairs:", len(miss_mc))

print("\nFirst few missing (maskclip):")
for x in miss_mc[:10]:
    print(x)

# %% [cell 16]
# ===== PATCH: fill missing polygons for v2 =====
import json, base64, requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
PAIRS = CKPT/"pairs_303w_v2.jsonl"
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"
ERR2  = CKPT/"b2_polygon_allpairs_v2_errors.jsonl"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_COVER_FRAC = 0.30

SYSTEM = (
    "你是视觉审计专家。任务：仅标注应忽略的“噪声/覆盖/水印/边框”区域，用于证据审计与Mask-CLIP。"
    "必须输出严格JSON，不要Markdown。"
)

USER_TMPL = f"""
给你两张图：query 与 candidate。我们只对 candidate 标注“噪声区域”（watermark/overlay_text/border等）。
你必须先判断噪声类型 noise_type，然后再给出至多3个多边形（LabelMe风格 points）。

强约束（必须遵守）：
1) 只标注噪声/覆盖/水印/边框等应忽略区域；不要标主体结构。
2) polygon 数量：0~3 个。
3) 每个 polygon 要尽量紧致。
4) 总覆盖面积必须 <= {MAX_COVER_FRAC:.2f}。若全图噪声：noise_type="global"，polygons=[]。
5) 坐标归一化到 [0,1]，顺时针，>=4点。
6) 每个 polygon 一句中文 reason_zh；并给全局 reason_zh。

noise_type: none|overlay_text|watermark|border|compression|global|unknown

输出严格 JSON：
{{
  "schema": "polygon_v2",
  "noise_type": "none|overlay_text|watermark|border|compression|global|unknown",
  "polygons": [{{"label":"noise","points":[[0.1,0.2],[...]],"reason_zh":"中文一句话"}}],
  "reason_zh": "中文一句话总结"
}}
""".strip()

def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    mime = "image/jpeg" if suf in [".jpg",".jpeg"] else "image/png" if suf==".png" else "image/webp" if suf==".webp" else "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower()=="json":
            txt="\n".join(lines[1:])
    return json.loads(txt)

def schema_check(d: dict):
    assert d.get("schema")=="polygon_v2"
    assert d.get("noise_type") in ["none","overlay_text","watermark","border","compression","global","unknown"]
    assert isinstance(d.get("polygons"), list) and len(d["polygons"])<=3
    if d["noise_type"]=="global":
        assert len(d["polygons"])==0
    for poly in d["polygons"]:
        assert poly.get("label")=="noise"
        pts = poly.get("points")
        assert isinstance(pts, list) and len(pts)>=4
        for x,y in pts:
            x=float(x); y=float(y)
            assert 0.0<=x<=1.0 and 0.0<=y<=1.0
        assert isinstance(poly.get("reason_zh",""), str)

def call_8b_polygon(q_path: Path, c_path: Path) -> dict:
    payload = {
        "model":"qwen3-vl-8b",
        "messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":USER_TMPL},
                {"type":"image_url","image_url":{"url":to_data_url(q_path)}},
                {"type":"image_url","image_url":{"url":to_data_url(c_path)}},
            ]}
        ],
        "temperature":0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)

def load_keys(path: Path):
    s=set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r=json.loads(line)
                    s.add((r["query_image_id"], r["candidate_image_id"]))
                except: pass
    return s

k_pairs = load_keys(PAIRS)
k_poly2 = load_keys(POLY2)
missing = sorted(list(k_pairs - k_poly2))
print("[MISSING polygons]", len(missing))
for x in missing:
    print(" -", x)

# index pairs file rows for quick lookup of missing
need=set(missing)
rows=[]
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        k=(r["query_image_id"], r["candidate_image_id"])
        if k in need:
            rows.append(r)

print("[FOUND rows]", len(rows))

ok=0; fail=0
with POLY2.open("a", encoding="utf-8") as fout, ERR2.open("a", encoding="utf-8") as ferr:
    for row in tqdm(rows, desc="Patch polygons v2", ncols=90):
        qid=row["query_image_id"]; cid=row["candidate_image_id"]
        try:
            q_path=Path(row["query_path"]); c_path=Path(row["candidate_path"])
            res=call_8b_polygon(q_path,c_path)
            schema_check(res)

            out={
                "schema":"b2_polygon_allpairs_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": row["query_path"],
                "candidate_path": row["candidate_path"],
                "noise_type": res["noise_type"],
                "polygons": res["polygons"],
                "reason": res["reason_zh"],
                "max_cover_frac": MAX_COVER_FRAC,
                "patched": True,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n")
            fout.flush()
            ok += 1
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"b2_polygon_allpairs_v2_patch_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False)+"\n")
            ferr.flush()
            fail += 1

print("[DONE] patch polygons v2")
print("[STATS] ok:", ok, "fail:", fail)

# %% [cell 17]
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from transformers import CLIPProcessor, CLIPModel

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
PAIRS = CKPT/"pairs_303w_v2.jsonl"
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"
MC2   = CKPT/"c_maskclip_303w_v2.jsonl"
ERR2  = CKPT/"c_maskclip_303w_v2_errors.jsonl"

LOCAL_CLIP = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(str(LOCAL_CLIP)).eval().to(device)
processor = CLIPProcessor.from_pretrained(str(LOCAL_CLIP))
print("[OK] CLIP loaded on", device)

def load_keys(path: Path):
    s=set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r=json.loads(line)
                    s.add((r["query_image_id"], r["candidate_image_id"]))
                except: pass
    return s

# compute missing keys
k_pairs = load_keys(PAIRS)
k_mc2   = load_keys(MC2)
missing = sorted(list(k_pairs - k_mc2))
print("[MISSING]", len(missing))
for x in missing:
    print(" -", x)

assert len(missing) == 3, "Expected 3 missing keys; if not, stop and inspect."

# load pairs -> paths
pair_rows={}
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        k=(r["query_image_id"], r["candidate_image_id"])
        pair_rows[k]=r

# load poly
poly={}
with POLY2.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        k=(r["query_image_id"], r["candidate_image_id"])
        poly[k]=r

def denorm_points(pts, w, h):
    return [(float(x)*w, float(y)*h) for x,y in pts]

def make_mask(img_size, polygons):
    w,h = img_size
    m = Image.new("L", (w,h), 0)
    dr = ImageDraw.Draw(m)
    for p in polygons:
        pts = denorm_points(p["points"], w, h)
        dr.polygon(pts, fill=255)
    return m

def apply_mask(img_rgb, mask_l):
    arr = np.array(img_rgb).copy()
    m = np.array(mask_l) > 0
    arr[m] = 0
    return Image.fromarray(arr)

@torch.no_grad()
def clip_sim(img1, img2):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0]*feats[1]).sum().item())

written = 0
with MC2.open("a", encoding="utf-8") as fout, ERR2.open("a", encoding="utf-8") as ferr:
    for k in missing:
        qid, cid = k
        try:
            pr = pair_rows[k]
            rr = poly[k]

            q_path = Path(pr["query_path"])
            c_path = Path(pr["candidate_path"])
            assert q_path.exists(), f"missing query: {q_path}"
            assert c_path.exists(), f"missing cand: {c_path}"

            q = Image.open(q_path).convert("RGB")
            cimg = Image.open(c_path).convert("RGB")

            clip_raw = clip_sim(q, cimg)

            noise_type = rr.get("noise_type")
            polys = rr.get("polygons", [])

            if noise_type == "global" or len(polys)==0:
                clip_masked = clip_raw
                explain = f"noise_type={noise_type}, polygons=0 -> no masking"
            else:
                mask = make_mask(cimg.size, polys)
                c_masked = apply_mask(cimg, mask)
                clip_masked = clip_sim(q, c_masked)
                explain = f"noise_type={noise_type}, polygons={len(polys)}"

            out = {
                "schema":"c_maskclip_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_raw": float(clip_raw),
                "clip_masked": float(clip_masked),
                "delta": float(clip_masked - clip_raw),
                "explain_zh": explain,
                "patched": True,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            print("[WRITE OK]", qid[:18], cid[:18], "delta=", out["delta"], "explain=", explain)

        except Exception as e:
            ferr.write(json.dumps({
                "schema":"c_maskclip_v2_patch_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            print("[WRITE FAIL]", qid[:18], cid[:18], "err=", str(e))

print("[DONE] patched maskclip v2 written:", written)

# re-check counts immediately
k_mc2_after = load_keys(MC2)
print("[COUNT AFTER] maskclip_v2:", len(k_mc2_after))

# %% [cell 18]
import json
import pandas as pd
from pathlib import Path

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
MC1 = CKPT/"c_maskclip_303w_v1.jsonl"
MC2 = CKPT/"c_maskclip_303w_v2.jsonl"

def load_delta(path):
    m={}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            k=(r["query_image_id"], r["candidate_image_id"])
            m[k]=float(r["delta"])
    return m

d1=load_delta(MC1)
d2=load_delta(MC2)

keys = sorted(list(set(d1.keys()) & set(d2.keys())))
s1=pd.Series([d1[k] for k in keys], name="delta_v1")
s2=pd.Series([d2[k] for k in keys], name="delta_v2")
diff=(s2-s1)

def q(s):
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

print("[DELTA v1]", q(s1))
print("[DELTA v2]", q(s2))
print("[DIFF v2-v1]", q(diff))

# tail comparisons: count of very negative deltas
thr1 = s1.quantile(0.01)
thr2 = s2.quantile(0.01)
print("\n[TAIL] v1 p01 threshold =", float(thr1), "count <=thr =", int((s1<=thr1).sum()))
print("[TAIL] v2 p01 threshold =", float(thr2), "count <=thr =", int((s2<=thr2).sum()))

# fixed absolute tail (more interpretable)
for T in [-0.30, -0.20, -0.10]:
    print(f"[ABS TAIL] delta<={T}: v1={(s1<=T).sum()} | v2={(s2<=T).sum()}")

# %% [cell 19]
import json
from pathlib import Path
import pandas as pd

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"

rows=[]
with POLY2.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        rows.append({
            "noise_type": r.get("noise_type"),
            "n_poly": len(r.get("polygons", []))
        })

df=pd.DataFrame(rows)
print("[noise_type dist]")
print(df["noise_type"].value_counts(dropna=False))
print("\n[n_poly dist]")
print(df["n_poly"].value_counts(dropna=False).sort_index())
print("\n[share polygons==0]", float((df["n_poly"]==0).mean()))

# %% [cell 20]
import json
from pathlib import Path
from collections import Counter

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
PAIRS = CKPT/"pairs_303w_v2.jsonl"

c = Counter()
n=0
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        p=str(r["candidate_path"]).lower().replace("/", "\\")
        if "\\archive\\303\\303u\\" in p:
            c["303u(clean)"] += 1
        elif "\\archive\\303\\303w\\" in p:
            c["303w(wm)"] += 1
        else:
            c["other"] += 1
        n += 1
print("[pairs lines]", n)
print(c)

# %% [cell 21]
import json, math
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT/"checkpoints"
OUTS = ROOT/"outputs"
OUTS.mkdir(parents=True, exist_ok=True)

POLY1 = CKPT/"b2_polygon_allpairs_v1.jsonl"
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"
MC1   = CKPT/"c_maskclip_303w_v1.jsonl"
MC2   = CKPT/"c_maskclip_303w_v2.jsonl"

assert POLY1.exists() and POLY2.exists() and MC1.exists() and MC2.exists()

def load_jsonl(path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def key_of(r):
    return (r["query_image_id"], r["candidate_image_id"])

def poly_cover_frac(candidate_path, polygons):
    # rasterize polygons into mask -> coverage fraction
    img = Image.open(candidate_path)
    w,h = img.size
    m = Image.new("L",(w,h),0)
    dr = ImageDraw.Draw(m)
    for p in polygons:
        pts = p.get("points") or p.get("polygon") or []
        if not pts: 
            continue
        pts_px = [(float(x)*w, float(y)*h) for x,y in pts]
        if len(pts_px) >= 3:
            dr.polygon(pts_px, fill=255)
    arr = np.array(m) > 0
    return float(arr.mean())

def poly_stats(poly_rows, version_name):
    out=[]
    for r in poly_rows:
        polys = r.get("polygons", [])
        n_poly = len(polys)
        cand = r.get("candidate_path")  # v1/v2 都是画在 candidate_path 上
        cov = None
        try:
            if cand and Path(cand).exists() and n_poly>0:
                cov = poly_cover_frac(cand, polys)
            else:
                cov = 0.0
        except:
            cov = None
        out.append({
            "version": version_name,
            "key": str(key_of(r)),
            "noise_type": r.get("noise_type"),
            "n_poly": n_poly,
            "cover_frac": cov,
        })
    return pd.DataFrame(out)

def maskclip_stats(mc_rows, version_name):
    out=[]
    for r in mc_rows:
        out.append({
            "version": version_name,
            "key": str(key_of(r)),
            "delta": float(r["delta"]),
            "clip_raw": float(r.get("clip_raw", np.nan)),
            "clip_masked": float(r.get("clip_masked", np.nan)),
        })
    return pd.DataFrame(out)

p1 = poly_stats(load_jsonl(POLY1), "v1")
p2 = poly_stats(load_jsonl(POLY2), "v2")
m1 = maskclip_stats(load_jsonl(MC1), "v1")
m2 = maskclip_stats(load_jsonl(MC2), "v2")

def summarize_poly(df):
    s={}
    s["n"] = int(df.shape[0])
    s["noise_type_dist"] = df["noise_type"].value_counts(dropna=False).to_dict()
    s["n_poly_dist"] = df["n_poly"].value_counts(dropna=False).sort_index().to_dict()
    s["cover_frac_desc"] = {
        "mean": float(np.nanmean(df["cover_frac"])),
        "p50": float(np.nanquantile(df["cover_frac"], 0.50)),
        "p90": float(np.nanquantile(df["cover_frac"], 0.90)),
        "p95": float(np.nanquantile(df["cover_frac"], 0.95)),
        "max": float(np.nanmax(df["cover_frac"])),
    }
    return s

def summarize_delta(df):
    d = df["delta"].astype(float)
    s={}
    s["n"] = int(d.shape[0])
    s["mean"] = float(d.mean())
    s["std"] = float(d.std(ddof=0))
    for q in [0.01,0.05,0.5,0.95,0.99]:
        s[f"p{int(q*100):02d}"] = float(d.quantile(q))
    s["min"] = float(d.min())
    s["max"] = float(d.max())
    s["abs_tail_counts"] = {t: int((d<=t).sum()) for t in [-0.10,-0.20,-0.30]}
    return s

summary = {
    "v1": {
        "polygon": summarize_poly(p1),
        "maskclip": summarize_delta(m1),
    },
    "v2": {
        "polygon": summarize_poly(p2),
        "maskclip": summarize_delta(m2),
    }
}

# export
p_all = pd.concat([p1,p2], ignore_index=True)
m_all = pd.concat([m1,m2], ignore_index=True)

p_all.to_csv(OUTS/"compare_v1_v2_polygon_stats.csv", index=False, encoding="utf-8-sig")
m_all.to_csv(OUTS/"compare_v1_v2_maskclip_stats.csv", index=False, encoding="utf-8-sig")
(OUTS/"summary_compare_v1_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("[OUT]", OUTS/"compare_v1_v2_polygon_stats.csv")
print("[OUT]", OUTS/"compare_v1_v2_maskclip_stats.csv")
print("[OUT]", OUTS/"summary_compare_v1_v2.json")
print("\n[SUMMARY]\n", json.dumps(summary, ensure_ascii=False, indent=2)[:1500], "...\n")

# %% [cell 22]
# ===== Cell A: v3 Polygon (conditional, target=noisy query_path) =====
import json, base64, requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
IN_PAIRS = CKPT/"pairs_303w_v2.jsonl"
OUT_POLY = CKPT/"b2_polygon_allpairs_v3.jsonl"
ERR_POLY = CKPT/"b2_polygon_allpairs_v3_errors.jsonl"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_COVER_FRAC = 0.25

SYSTEM = (
    "你是视觉审计专家。任务：仅在确实存在噪声时，标注应忽略的噪声区域。"
    "若不存在噪声，必须明确返回 noise_type=none 且 polygons=[]."
    "只输出严格JSON，不要Markdown。"
)

USER = f"""
给你一张带噪声的图像（noisy image）。你需要判断是否存在应忽略的噪声区域（如水印、叠加文字、边框、压缩伪影）。
规则：
1) 若不存在噪声：noise_type="none"，polygons=[]；
2) 若存在噪声：只标注噪声区域（不要标主体），polygons 0~3 个；
3) polygon 必须紧致；总覆盖面积 <= {MAX_COVER_FRAC:.2f}；
4) 坐标归一化到 [0,1]，顺时针，>=4点；
5) 给出中文一句话 reason_zh。

noise_type: none|overlay_text|watermark|border|compression|unknown

输出严格 JSON：
{{
  "schema": "polygon_v3",
  "noise_type": "none|overlay_text|watermark|border|compression|unknown",
  "polygons": [{{"label":"noise","points":[[0.1,0.2],[...]],"reason_zh":"中文一句话"}}],
  "reason_zh": "中文一句话总结"
}}
""".strip()

def to_data_url(p: Path) -> str:
    suf = p.suffix.lower()
    mime = "image/jpeg" if suf in [".jpg",".jpeg"] else "image/png" if suf==".png" else "image/webp"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower()=="json":
            txt="\n".join(lines[1:])
    return json.loads(txt)

def schema_check(d):
    assert d.get("schema")=="polygon_v3"
    nt = d.get("noise_type")
    assert nt in ["none","overlay_text","watermark","border","compression","unknown"]
    polys = d.get("polygons")
    assert isinstance(polys, list) and len(polys)<=3
    if nt=="none":
        assert len(polys)==0
    for p in polys:
        pts = p["points"]
        assert isinstance(pts,list) and len(pts)>=4
        for x,y in pts:
            x=float(x); y=float(y)
            assert 0<=x<=1 and 0<=y<=1

# resume
done=set()
if OUT_POLY.exists():
    with OUT_POLY.open("r", encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            done.add((r["query_image_id"], r["candidate_image_id"]))
print("[INFO] already done:", len(done))

ok=0; fail=0
with IN_PAIRS.open("r", encoding="utf-8") as fin, \
     OUT_POLY.open("a", encoding="utf-8") as fout, \
     ERR_POLY.open("a", encoding="utf-8") as ferr:
    for line in tqdm(fin, desc="v3 polygon", ncols=90):
        row=json.loads(line)
        k=(row["query_image_id"], row["candidate_image_id"])
        if k in done: continue
        try:
            noisy = Path(row["query_path"])  # 303w
            assert noisy.exists()
            payload={
                "model":"qwen3-vl-8b",
                "messages":[
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":[
                        {"type":"text","text":USER},
                        {"type":"image_url","image_url":{"url":to_data_url(noisy)}},
                    ]}
                ],
                "temperature":0.0,
            }
            r = requests.post(LLM_URL, json=payload, timeout=240)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            res = extract_json(raw)
            schema_check(res)

            out={
                "schema":"b2_polygon_allpairs_v3",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "noisy_path": row["query_path"],
                "noise_type": res["noise_type"],
                "polygons": res["polygons"],
                "reason": res["reason_zh"],
                "max_cover_frac": MAX_COVER_FRAC,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n"); fout.flush()
            ok+=1; done.add(k)
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"b2_polygon_allpairs_v3_error",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "error": str(e),
            }, ensure_ascii=False)+"\n"); ferr.flush()
            fail+=1

print("[DONE] v3 polygon"); print("[STATS] ok:",ok,"fail:",fail)

# %% [cell 23]
# ===== Cell B: v3 Overlay =====
import json
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY3 = CKPT/"b2_polygon_allpairs_v3.jsonl"
VIZ = Path(r"<PROTOTYPE303_ROOT>\viz_v3\pairs")
VIZ.mkdir(parents=True, exist_ok=True)

def denorm(pts,w,h): return [(x*w,y*h) for x,y in pts]

rows=[]
with POLY3.open("r", encoding="utf-8") as f:
    for line in f: rows.append(json.loads(line))

ok=0
for r in tqdm(rows, desc="v3 overlay", ncols=90):
    qid=r["query_image_id"].split("sha256:")[1][:17]
    cid=r["candidate_image_id"].split("sha256:")[1][:17]
    d=VIZ/f"{qid}__{cid}"
    d.mkdir(parents=True, exist_ok=True)

    img=Image.open(r["noisy_path"]).convert("RGB")
    w,h=img.size
    draw=ImageDraw.Draw(img, "RGBA")
    for p in r["polygons"]:
        pts=denorm(p["points"],w,h)
        draw.polygon(pts, outline=(255,0,0,255), fill=(255,0,0,80))
    img.save(d/"query_overlay.png")

    meta={k:r[k] for k in ["noise_type","reason","noisy_path","query_image_id","candidate_image_id"]}
    (d/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ok+=1

print("[DONE] v3 overlay ok:", ok)

# %% [cell 24]
# ===== Cell C: v3 Mask-CLIP (conditional) =====
import json, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY3 = CKPT/"b2_polygon_allpairs_v3.jsonl"
PAIRS = CKPT/"pairs_303w_v2.jsonl"
OUT = CKPT/"c_maskclip_303w_v3.jsonl"
ERR = CKPT/"c_maskclip_303w_v3_errors.jsonl"

LOCAL_CLIP = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
device="cuda" if torch.cuda.is_available() else "cpu"
model=CLIPModel.from_pretrained(str(LOCAL_CLIP)).eval().to(device)
proc=CLIPProcessor.from_pretrained(str(LOCAL_CLIP))
print("[OK] CLIP on",device)

pairs={}
with PAIRS.open("r",encoding="utf-8") as f:
    for l in f:
        r=json.loads(l); pairs[(r["query_image_id"],r["candidate_image_id"])]=r

def mask_img(img, polys):
    w,h=img.size
    m=Image.new("L",(w,h),0); d=ImageDraw.Draw(m)
    for p in polys:
        pts=[(x*w,y*h) for x,y in p["points"]]
        d.polygon(pts, fill=255)
    arr=np.array(img); arr[np.array(m)>0]=0
    return Image.fromarray(arr)

@torch.no_grad()
def sim(a,b):
    x=proc(images=[a,b], return_tensors="pt").to(device)
    f=model.get_image_features(**x); f=f/f.norm(dim=-1,keepdim=True)
    return float((f[0]*f[1]).sum())

# resume
done=set()
if OUT.exists():
    with OUT.open("r",encoding="utf-8") as f:
        for l in f:
            r=json.loads(l); done.add((r["query_image_id"],r["candidate_image_id"]))
print("[INFO] already done:", len(done))

ok=0; fail=0
with POLY3.open("r",encoding="utf-8") as fin, \
     OUT.open("a",encoding="utf-8") as fout, \
     ERR.open("a",encoding="utf-8") as ferr:
    for l in tqdm(fin, desc="v3 mask-clip", ncols=90):
        r=json.loads(l)
        k=(r["query_image_id"], r["candidate_image_id"])
        if k in done: continue
        try:
            pr=pairs[k]
            noisy=Image.open(r["noisy_path"]).convert("RGB")
            clean=Image.open(pr["candidate_path"]).convert("RGB")
            raw=sim(noisy, clean)
            if len(r["polygons"])==0:
                masked=raw; delta=0.0; explain="no noise -> no mask"
            else:
                noisy_m=mask_img(noisy, r["polygons"])
                masked=sim(noisy_m, clean); delta=masked-raw
                explain=f"masked {len(r['polygons'])} noise regions"
            out={
                "schema":"c_maskclip_v3",
                "query_image_id": k[0],
                "candidate_image_id": k[1],
                "clip_raw": raw,
                "clip_masked": masked,
                "delta": delta,
                "explain_zh": explain,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n"); fout.flush()
            ok+=1; done.add(k)
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"c_maskclip_v3_error",
                "query_image_id": k[0],
                "candidate_image_id": k[1],
                "error": str(e),
            }, ensure_ascii=False)+"\n"); ferr.flush()
            fail+=1

print("[DONE] v3 mask-clip"); print("[STATS] ok:",ok,"fail:",fail)

# %% [cell 25]
# ===== Cell D: compare v1/v2/v3 =====
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT/"checkpoints"
OUTS = ROOT/"outputs"
OUTS.mkdir(parents=True, exist_ok=True)

# --- inputs ---
POLY1 = CKPT/"b2_polygon_allpairs_v1.jsonl"
POLY2 = CKPT/"b2_polygon_allpairs_v2.jsonl"
POLY3 = CKPT/"b2_polygon_allpairs_v3.jsonl"

MC1   = CKPT/"c_maskclip_303w_v1.jsonl"
MC2   = CKPT/"c_maskclip_303w_v2.jsonl"
MC3   = CKPT/"c_maskclip_303w_v3.jsonl"

PAIRS = CKPT/"pairs_303w_v2.jsonl"

for p in [POLY1,POLY2,POLY3,MC1,MC2,MC3,PAIRS]:
    assert p.exists(), f"Missing: {p}"

def load_jsonl(path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def key_of(r):
    return (r["query_image_id"], r["candidate_image_id"])

pairs={}
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        pairs[key_of(r)] = r

def poly_cover_frac(img_path, polygons):
    img = Image.open(img_path)
    w,h = img.size
    m = Image.new("L",(w,h),0)
    dr = ImageDraw.Draw(m)
    for p in polygons:
        pts = p.get("points") or []
        if not pts: 
            continue
        pts_px = [(float(x)*w, float(y)*h) for x,y in pts]
        if len(pts_px) >= 3:
            dr.polygon(pts_px, fill=255)
    arr = np.array(m) > 0
    return float(arr.mean())

def poly_df(rows, version):
    out=[]
    for r in rows:
        k = key_of(r)
        # v1/v2: candidate_path as target (historical)
        # v3: noisy_path as target (correct)
        if version in ["v1","v2"]:
            target = pairs[k]["candidate_path"]
            noise_type = r.get("noise_type")  # v1 may be null
            polys = r.get("polygons", []) if isinstance(r.get("polygons"), list) else []
        else:
            target = r.get("noisy_path")
            noise_type = r.get("noise_type")
            polys = r.get("polygons", [])
        n_poly = len(polys)
        cov = 0.0
        try:
            if n_poly>0 and target and Path(target).exists():
                cov = poly_cover_frac(target, polys)
        except:
            cov = np.nan
        out.append({
            "version": version,
            "query_image_id": k[0],
            "candidate_image_id": k[1],
            "noise_type": noise_type,
            "n_poly": n_poly,
            "cover_frac": cov,
        })
    return pd.DataFrame(out)

def mc_df(rows, version):
    out=[]
    for r in rows:
        k=key_of(r)
        out.append({
            "version": version,
            "query_image_id": k[0],
            "candidate_image_id": k[1],
            "delta": float(r["delta"]),
            "clip_raw": float(r.get("clip_raw", np.nan)),
            "clip_masked": float(r.get("clip_masked", np.nan)),
        })
    return pd.DataFrame(out)

p1 = poly_df(load_jsonl(POLY1), "v1")
p2 = poly_df(load_jsonl(POLY2), "v2")
p3 = poly_df(load_jsonl(POLY3), "v3")

m1 = mc_df(load_jsonl(MC1), "v1")
m2 = mc_df(load_jsonl(MC2), "v2")
m3 = mc_df(load_jsonl(MC3), "v3")

p_all = pd.concat([p1,p2,p3], ignore_index=True)
m_all = pd.concat([m1,m2,m3], ignore_index=True)

def summarize_poly(df):
    s={}
    s["n"] = int(df.shape[0])
    s["noise_type_dist"] = df["noise_type"].value_counts(dropna=False).to_dict()
    s["n_poly_dist"] = df["n_poly"].value_counts(dropna=False).sort_index().to_dict()
    cf = df["cover_frac"].astype(float)
    s["cover_frac_desc"] = {
        "mean": float(np.nanmean(cf)),
        "p50": float(np.nanquantile(cf, 0.50)),
        "p90": float(np.nanquantile(cf, 0.90)),
        "p95": float(np.nanquantile(cf, 0.95)),
        "max": float(np.nanmax(cf)),
    }
    s["share_empty_polygons"] = float((df["n_poly"]==0).mean())
    return s

def summarize_delta(df):
    d = df["delta"].astype(float)
    s={}
    s["n"]=int(d.shape[0])
    s["mean"]=float(d.mean())
    s["std"]=float(d.std(ddof=0))
    for q in [0.01,0.05,0.5,0.95,0.99]:
        s[f"p{int(q*100):02d}"]=float(d.quantile(q))
    s["min"]=float(d.min()); s["max"]=float(d.max())
    s["abs_tail_counts"] = {t:int((d<=t).sum()) for t in [-0.10,-0.20,-0.30]}
    return s

summary = {
    "v1": {"polygon": summarize_poly(p1), "maskclip": summarize_delta(m1)},
    "v2": {"polygon": summarize_poly(p2), "maskclip": summarize_delta(m2)},
    "v3": {"polygon": summarize_poly(p3), "maskclip": summarize_delta(m3)},
}

# export
p_all.to_csv(OUTS/"compare_v1_v2_v3_polygon_stats.csv", index=False, encoding="utf-8-sig")
m_all.to_csv(OUTS/"compare_v1_v2_v3_maskclip_stats.csv", index=False, encoding="utf-8-sig")
(OUTS/"summary_compare_v1_v2_v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# quick audit samples for v3
# export 20 samples per noise_type (by count order)
aud = {}
p3_nt = p3.copy()
p3_nt["qid_short"] = p3_nt["query_image_id"].apply(lambda x: x.split("sha256:")[1][:17])
p3_nt["cid_short"] = p3_nt["candidate_image_id"].apply(lambda x: x.split("sha256:")[1][:17])
for nt, g in p3_nt.groupby("noise_type"):
    aud[nt] = g.head(20)[["qid_short","cid_short","n_poly","cover_frac"]].to_dict(orient="records")
(OUTS/"audit_v3_noise_type_samples.json").write_text(json.dumps(aud, ensure_ascii=False, indent=2), encoding="utf-8")

print("[OUT]", OUTS/"compare_v1_v2_v3_polygon_stats.csv")
print("[OUT]", OUTS/"compare_v1_v2_v3_maskclip_stats.csv")
print("[OUT]", OUTS/"summary_compare_v1_v2_v3.json")
print("[OUT]", OUTS/"audit_v3_noise_type_samples.json")
print("\n[SUMMARY]\n", json.dumps(summary, ensure_ascii=False, indent=2)[:1600], "...\n")

# %% [cell 26]
# ===== Cell A: v3.1 Polygon (conditional + allow none) =====
import json, base64, requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
IN_PAIRS = CKPT / "pairs_303w_v2.jsonl"

OUT_POLY = CKPT / "b2_polygon_allpairs_v3_1.jsonl"
ERR_POLY = CKPT / "b2_polygon_allpairs_v3_1_errors.jsonl"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ------- v3.1 knobs -------
MAX_COVER_FRAC = 0.25      # 总覆盖面积上限（软约束：模型遵守）
TAU_NOISE = 0.55           # 噪声置信门控阈值：<tau 强制 none
MAX_POLY = 3               # 最多 3 个 polygon
TEMP = 0.0

assert IN_PAIRS.exists(), f"Missing: {IN_PAIRS}"

SYSTEM = (
    "你是视觉审计专家。任务：识别并标注图像中应忽略的噪声区域（如水印/叠加文字/边框/压缩伪影）。"
    "注意：噪声并非一定存在。若你不确信存在噪声，必须返回 noise_type=none 且 polygons=[]。"
    "只输出严格 JSON，不要输出 Markdown。"
)

USER = f"""
你将看到一张 noisy image（可能包含水印/文字/边框/压缩噪声，也可能完全没有噪声）。
请判断是否存在应忽略的噪声区域，并输出严格 JSON。

评分锚点：noise_confidence ∈ [0,1] 表示你对“存在噪声”的确信程度。
- 若 noise_confidence < {TAU_NOISE:.2f}：你必须输出 noise_type="none" 且 polygons=[]
- 若 noise_confidence >= {TAU_NOISE:.2f}：你可以输出噪声类型，并标注 0~{MAX_POLY} 个 polygon（通常 1 个即可）

约束：
1) 只标注噪声区域，不要标主体/关键结构；
2) polygons 最多 {MAX_POLY} 个，尽量紧致；
3) 总覆盖面积尽量 <= {MAX_COVER_FRAC:.2f}（超过也可以，但需在 reason_zh 中解释为什么）；
4) 坐标归一化到 [0,1]，顺时针，>=4点；
5) 每个 polygon 给一句中文 reason_zh；再给整体 reason_zh 一句总结。

noise_type 取值：
none|overlay_text|watermark|border|compression|unknown

输出 JSON schema（必须完全符合）：
{{
  "schema": "polygon_v3_1",
  "noise_confidence": 0.0,
  "noise_type": "none|overlay_text|watermark|border|compression|unknown",
  "polygons": [
    {{"label":"noise","points":[[0.1,0.2],[...]],"reason_zh":"中文一句话"}}
  ],
  "reason_zh": "中文一句话总结"
}}
""".strip()

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
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    return json.loads(txt)

def schema_check(d: dict):
    assert d.get("schema") == "polygon_v3_1"
    conf = float(d.get("noise_confidence"))
    assert 0.0 <= conf <= 1.0
    nt = d.get("noise_type")
    assert nt in ["none", "overlay_text", "watermark", "border", "compression", "unknown"]
    polys = d.get("polygons")
    assert isinstance(polys, list) and len(polys) <= MAX_POLY

    # gate enforcement (hard)
    if conf < TAU_NOISE:
        assert nt == "none"
        assert len(polys) == 0

    for p in polys:
        assert p.get("label") == "noise"
        pts = p.get("points")
        assert isinstance(pts, list) and len(pts) >= 4
        for x, y in pts:
            x = float(x); y = float(y)
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        rz = p.get("reason_zh")
        assert isinstance(rz, str) and len(rz) > 0

    rz2 = d.get("reason_zh")
    assert isinstance(rz2, str) and len(rz2) > 0

# resume
done=set()
if OUT_POLY.exists():
    with OUT_POLY.open("r", encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            done.add((r["query_image_id"], r["candidate_image_id"]))
print("[INFO] already done:", len(done))

ok=0; fail=0
with IN_PAIRS.open("r", encoding="utf-8") as fin, \
     OUT_POLY.open("a", encoding="utf-8") as fout, \
     ERR_POLY.open("a", encoding="utf-8") as ferr:
    for line in tqdm(fin, desc="v3.1 polygon", ncols=90):
        row=json.loads(line)
        k=(row["query_image_id"], row["candidate_image_id"])
        if k in done:
            continue
        try:
            noisy = Path(row["query_path"])  # 303w noisy
            assert noisy.exists(), f"missing noisy query_path: {noisy}"

            payload={
                "model":"qwen3-vl-8b",
                "messages":[
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":[
                        {"type":"text","text":USER},
                        {"type":"image_url","image_url":{"url":to_data_url(noisy)}},
                    ]},
                ],
                "temperature": TEMP,
            }
            r = requests.post(LLM_URL, json=payload, timeout=240)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            res = extract_json(raw)
            schema_check(res)

            out={
                "schema":"b2_polygon_allpairs_v3_1",
                "query_image_id": row["query_image_id"],
                "candidate_image_id": row["candidate_image_id"],
                "noisy_path": row["query_path"],
                "noise_confidence": float(res["noise_confidence"]),
                "noise_type": res["noise_type"],
                "polygons": res["polygons"],
                "reason": res["reason_zh"],
                "max_cover_frac": MAX_COVER_FRAC,
                "tau_noise": TAU_NOISE,
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n"); fout.flush()
            ok += 1
            done.add(k)
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"b2_polygon_allpairs_v3_1_error",
                "query_image_id": row.get("query_image_id"),
                "candidate_image_id": row.get("candidate_image_id"),
                "error": str(e),
            }, ensure_ascii=False)+"\n"); ferr.flush()
            fail += 1

print("[DONE] v3.1 polygon")
print("[STATS] ok:", ok, "fail:", fail)
print("[OUT]", OUT_POLY, "bytes=", OUT_POLY.stat().st_size if OUT_POLY.exists() else -1)
print("[ERR]", ERR_POLY, "bytes=", ERR_POLY.stat().st_size if ERR_POLY.exists() else -1)

# %% [cell 27]
# ===== Cell B: v3.1 Overlay (supports empty polygons) =====
import json
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY = CKPT / "b2_polygon_allpairs_v3_1.jsonl"

VIZ = Path(r"<PROTOTYPE303_ROOT>\viz_v3_1\pairs")
VIZ.mkdir(parents=True, exist_ok=True)

assert POLY.exists(), f"Missing: {POLY}"

def denorm(pts, w, h):
    return [(float(x)*w, float(y)*h) for x,y in pts]

rows=[]
with POLY.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

ok=0
for r in tqdm(rows, desc="v3.1 overlay", ncols=90):
    qid=r["query_image_id"].split("sha256:")[1][:17]
    cid=r["candidate_image_id"].split("sha256:")[1][:17]
    d=VIZ/f"{qid}__{cid}"
    d.mkdir(parents=True, exist_ok=True)

    img=Image.open(r["noisy_path"]).convert("RGB")
    w,h=img.size
    draw=ImageDraw.Draw(img, "RGBA")

    polys = r.get("polygons", [])
    for p in polys:
        pts=p.get("points", [])
        if len(pts) >= 3:
            pts_px=denorm(pts,w,h)
            draw.polygon(pts_px, outline=(255,0,0,255), fill=(255,0,0,80))

    img.save(d/"query_overlay.png")

    meta={
        "schema":"viz_meta_v3_1",
        "query_image_id": r["query_image_id"],
        "candidate_image_id": r["candidate_image_id"],
        "noisy_path": r["noisy_path"],
        "noise_confidence": r.get("noise_confidence"),
        "noise_type": r.get("noise_type"),
        "n_poly": len(polys),
        "reason": r.get("reason"),
        "tau_noise": r.get("tau_noise"),
    }
    (d/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ok += 1

print("[DONE] v3.1 overlay ok:", ok)
print("[VIZ_ROOT]", VIZ)

# %% [cell 28]
# ===== Cell C: v3.1 Mask-CLIP (conditional) =====
import json, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

CKPT = Path(r"<PROTOTYPE303_ROOT>\checkpoints")
POLY = CKPT/"b2_polygon_allpairs_v3_1.jsonl"
PAIRS = CKPT/"pairs_303w_v2.jsonl"

OUT = CKPT/"c_maskclip_303w_v3_1.jsonl"
ERR = CKPT/"c_maskclip_303w_v3_1_errors.jsonl"

LOCAL_CLIP = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert POLY.exists() and PAIRS.exists()

device="cuda" if torch.cuda.is_available() else "cpu"
model=CLIPModel.from_pretrained(str(LOCAL_CLIP)).eval().to(device)
proc=CLIPProcessor.from_pretrained(str(LOCAL_CLIP))
print("[OK] CLIP on", device)

pairs={}
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        pairs[(r["query_image_id"], r["candidate_image_id"])] = r

def mask_img(img, polys):
    w,h=img.size
    m=Image.new("L",(w,h),0); d=ImageDraw.Draw(m)
    for p in polys:
        pts=[(float(x)*w, float(y)*h) for x,y in p["points"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    arr=np.array(img)
    arr[np.array(m)>0]=0
    return Image.fromarray(arr)

@torch.no_grad()
def sim(a,b):
    x=proc(images=[a,b], return_tensors="pt").to(device)
    f=model.get_image_features(**x)
    f=f/f.norm(dim=-1, keepdim=True)
    return float((f[0]*f[1]).sum())

# resume
done=set()
if OUT.exists():
    with OUT.open("r", encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            done.add((r["query_image_id"], r["candidate_image_id"]))
print("[INFO] already done:", len(done))

ok=0; fail=0
with POLY.open("r", encoding="utf-8") as fin, \
     OUT.open("a", encoding="utf-8") as fout, \
     ERR.open("a", encoding="utf-8") as ferr:
    for line in tqdm(fin, desc="v3.1 mask-clip", ncols=90):
        r=json.loads(line)
        k=(r["query_image_id"], r["candidate_image_id"])
        if k in done:
            continue
        try:
            pr=pairs[k]
            noisy=Image.open(r["noisy_path"]).convert("RGB")          # 303w
            clean=Image.open(pr["candidate_path"]).convert("RGB")     # 303u

            raw=sim(noisy, clean)
            polys = r.get("polygons", [])

            if len(polys)==0:
                masked=raw
                delta=0.0
                explain="noise_confidence below tau -> no mask"
            else:
                noisy_m=mask_img(noisy, polys)
                masked=sim(noisy_m, clean)
                delta=masked-raw
                explain=f"masked {len(polys)} noise polygons"

            out={
                "schema":"c_maskclip_v3_1",
                "query_image_id": k[0],
                "candidate_image_id": k[1],
                "clip_raw": raw,
                "clip_masked": masked,
                "delta": delta,
                "explain_zh": explain,
                "noise_confidence": float(r.get("noise_confidence", 0.0)),
                "tau_noise": float(r.get("tau_noise", 0.0)),
                "noise_type": r.get("noise_type"),
            }
            fout.write(json.dumps(out, ensure_ascii=False)+"\n"); fout.flush()
            ok += 1
            done.add(k)
        except Exception as e:
            ferr.write(json.dumps({
                "schema":"c_maskclip_v3_1_error",
                "query_image_id": k[0],
                "candidate_image_id": k[1],
                "error": str(e),
            }, ensure_ascii=False)+"\n"); ferr.flush()
            fail += 1

print("[DONE] v3.1 mask-clip")
print("[STATS] ok:", ok, "fail:", fail)
print("[OUT]", OUT, "bytes=", OUT.stat().st_size if OUT.exists() else -1)
print("[ERR]", ERR, "bytes=", ERR.stat().st_size if ERR.exists() else -1)

# %% [cell 29]
# ===== Cell D' : v1/v2/v3/v3.1 summary + comparison tables =====
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUTS = ROOT / "outputs"
OUTS.mkdir(parents=True, exist_ok=True)

PAIRS = CKPT / "pairs_303w_v2.jsonl"
assert PAIRS.exists(), f"Missing: {PAIRS}"

FILES = {
    "v1": {
        "poly": CKPT / "b2_polygon_allpairs_v1.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v1.jsonl",
        "poly_target": "candidate_path",  # historical: v1/v2 polygon was on clean
    },
    "v2": {
        "poly": CKPT / "b2_polygon_allpairs_v2.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v2.jsonl",
        "poly_target": "candidate_path",
    },
    "v3": {
        "poly": CKPT / "b2_polygon_allpairs_v3.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v3.jsonl",
        "poly_target": "noisy_path",
    },
    "v3_1": {
        "poly": CKPT / "b2_polygon_allpairs_v3_1.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v3_1.jsonl",
        "poly_target": "noisy_path",
    }
}

for v, d in FILES.items():
    assert d["poly"].exists(), f"Missing {v} poly: {d['poly']}"
    assert d["mc"].exists(),   f"Missing {v} mc: {d['mc']}"

def load_jsonl(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def key_of(r): 
    return (r["query_image_id"], r["candidate_image_id"])

# load pairs map
pairs={}
with PAIRS.open("r", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        pairs[key_of(r)] = r

def poly_cover_frac(img_path, polygons):
    img = Image.open(img_path)
    w,h = img.size
    m = Image.new("L",(w,h),0)
    dr = ImageDraw.Draw(m)
    for p in polygons:
        pts = p.get("points") or []
        if len(pts) < 3:
            continue
        pts_px = [(float(x)*w, float(y)*h) for x,y in pts]
        dr.polygon(pts_px, fill=255)
    arr = np.array(m) > 0
    return float(arr.mean())

def poly_df(version: str):
    rows = load_jsonl(FILES[version]["poly"])
    target_mode = FILES[version]["poly_target"]  # candidate_path or noisy_path
    out=[]
    for r in rows:
        k = key_of(r)
        polys = r.get("polygons", [])
        n_poly = len(polys) if isinstance(polys, list) else 0
        noise_type = r.get("noise_type", None)
        noise_conf = r.get("noise_confidence", None)  # v3.1 only

        # choose target image for cover fraction
        if target_mode == "candidate_path":
            tgt = pairs[k].get("candidate_path")
        else:
            tgt = r.get("noisy_path") or pairs[k].get("query_path")

        cov = 0.0
        try:
            if n_poly > 0 and tgt and Path(tgt).exists():
                cov = poly_cover_frac(tgt, polys)
        except:
            cov = np.nan

        out.append({
            "version": version,
            "query_image_id": k[0],
            "candidate_image_id": k[1],
            "noise_type": noise_type,
            "noise_confidence": noise_conf,
            "n_poly": n_poly,
            "cover_frac": cov,
        })
    return pd.DataFrame(out)

def mc_df(version: str):
    rows = load_jsonl(FILES[version]["mc"])
    out=[]
    for r in rows:
        k = key_of(r)
        out.append({
            "version": version,
            "query_image_id": k[0],
            "candidate_image_id": k[1],
            "delta": float(r["delta"]),
            "clip_raw": float(r.get("clip_raw", np.nan)),
            "clip_masked": float(r.get("clip_masked", np.nan)),
        })
    return pd.DataFrame(out)

def summarize_poly(df: pd.DataFrame):
    s={}
    s["n"] = int(df.shape[0])
    s["noise_type_dist"] = df["noise_type"].value_counts(dropna=False).to_dict()
    s["n_poly_dist"] = df["n_poly"].value_counts(dropna=False).sort_index().to_dict()
    cf = df["cover_frac"].astype(float)
    s["cover_frac_desc"] = {
        "mean": float(np.nanmean(cf)),
        "p50": float(np.nanquantile(cf, 0.50)),
        "p90": float(np.nanquantile(cf, 0.90)),
        "p95": float(np.nanquantile(cf, 0.95)),
        "max": float(np.nanmax(cf)),
    }
    s["share_empty_polygons"] = float((df["n_poly"]==0).mean())
    # extra for v3.1: noise_conf stats if present
    if df["noise_confidence"].notna().any():
        nc = df["noise_confidence"].dropna().astype(float)
        s["noise_conf_desc"] = {
            "mean": float(nc.mean()),
            "p50": float(nc.quantile(0.50)),
            "p90": float(nc.quantile(0.90)),
            "min": float(nc.min()),
            "max": float(nc.max()),
        }
    return s

def summarize_delta(df: pd.DataFrame):
    d = df["delta"].astype(float)
    s={}
    s["n"]=int(d.shape[0])
    s["mean"]=float(d.mean())
    s["std"]=float(d.std(ddof=0))
    for q in [0.01,0.05,0.5,0.95,0.99]:
        s[f"p{int(q*100):02d}"]=float(d.quantile(q))
    s["min"]=float(d.min()); s["max"]=float(d.max())
    s["abs_tail_counts"] = {str(t): int((d<=t).sum()) for t in [-0.10,-0.20,-0.30]}
    return s

# build dfs
p_all = pd.concat([poly_df(v) for v in ["v1","v2","v3","v3_1"]], ignore_index=True)
m_all = pd.concat([mc_df(v)   for v in ["v1","v2","v3","v3_1"]], ignore_index=True)

summary = {}
for v in ["v1","v2","v3","v3_1"]:
    pv = p_all[p_all["version"]==v].copy()
    mv = m_all[m_all["version"]==v].copy()
    summary[v] = {
        "polygon": summarize_poly(pv),
        "maskclip": summarize_delta(mv),
    }

# export detailed csv
p_all.to_csv(OUTS/"compare_v1_v2_v3_v3_1_polygon_stats.csv", index=False, encoding="utf-8-sig")
m_all.to_csv(OUTS/"compare_v1_v2_v3_v3_1_maskclip_stats.csv", index=False, encoding="utf-8-sig")

# export summary json
(OUTS/"summary_compare_v1_v2_v3_v3_1.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---- 3-criteria table (303 stage) ----
# ① Recall@K: not distinguishable here (same candidate set); mark as N/A
# ② Dispute efficiency: use "evidence hallucination risk" proxy:
#    - share_empty_polygons (should be >0 for v3.1 ideally)
#    - cover_frac p90 (smaller = more localized)
# ③ Boundary coverage: delta tails and std
def get(d, path, default=None):
    cur=d
    for k in path:
        if k not in cur: return default
        cur=cur[k]
    return cur

rows=[]
for v in ["v1","v2","v3","v3_1"]:
    s = summary[v]
    rows.append({
        "version": v,
        "Recall@K (303 stage)": "N/A (fixed candidate set)",
        "Dispute proxy: share_empty_polygons": get(s, ["polygon","share_empty_polygons"]),
        "Dispute proxy: cover_frac_p90": get(s, ["polygon","cover_frac_desc","p90"]),
        "Boundary: delta_std": get(s, ["maskclip","std"]),
        "Boundary: tail<=-0.1": get(s, ["maskclip","abs_tail_counts","-0.1"]),
        "Boundary: tail<=-0.2": get(s, ["maskclip","abs_tail_counts","-0.2"]),
        "Boundary: tail<=-0.3": get(s, ["maskclip","abs_tail_counts","-0.3"]),
        "Boundary: delta_mean": get(s, ["maskclip","mean"]),
    })

table = pd.DataFrame(rows)
table.to_csv(OUTS/"table_3criteria_v1_v2_v3_v3_1.csv", index=False, encoding="utf-8-sig")

print("[OUT]", OUTS/"summary_compare_v1_v2_v3_v3_1.json")
print("[OUT]", OUTS/"compare_v1_v2_v3_v3_1_polygon_stats.csv")
print("[OUT]", OUTS/"compare_v1_v2_v3_v3_1_maskclip_stats.csv")
print("[OUT]", OUTS/"table_3criteria_v1_v2_v3_v3_1.csv")
print("\n[SUMMARY]\n", json.dumps(summary, ensure_ascii=False, indent=2)[:1800], "...\n")
print("\n[TABLE]\n", table)

# %% [cell 30]
# ===== Cell 1: v4 Polygon (explicit YES/NO) =====
import json, base64, requests
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"

IN_PAIRS = CKPT / "pairs_303w_v2.jsonl"
OUT_POLY = CKPT / "b2_polygon_allpairs_v4.jsonl"
ERR_POLY = CKPT / "b2_polygon_allpairs_v4_errors.jsonl"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

MAX_POLY = 3
TAU_NOISE = 0.70
TEMP = 0.0

SYSTEM = (
    "你是视觉审计专家。你必须先判断图像中是否存在显式噪声。"
    "噪声并非必然存在。若不确信存在噪声，必须明确回答 NO，并返回空 polygons。"
    "只输出严格 JSON。"
)

USER = f"""
请先判断是否存在显式噪声（YES/NO），再决定是否画噪声 polygon。

规则：
- 若 noise_decision=NO，polygons 必须是 []
- 若 noise_decision=YES 但 noise_confidence < {TAU_NOISE}，也必须视为 NO

输出 JSON：
{{
  "schema": "polygon_v4",
  "noise_decision": "YES|NO",
  "noise_confidence": 0.0,
  "noise_type": "none|overlay_text|watermark|border|compression|unknown",
  "polygons": [
    {{"label":"noise","points":[[0.1,0.2],[...]],"reason_zh":"一句中文理由"}}
  ],
  "reason_zh": "一句中文总结"
}}
""".strip()

def to_data_url(p: Path):
    b64 = base64.b64encode(p.read_bytes()).decode()
    mime = "image/png" if p.suffix.lower()==".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def extract_json(txt):
    txt = txt.strip().strip("`")
    if txt.startswith("json"):
        txt = "\n".join(txt.splitlines()[1:])
    return json.loads(txt)

def schema_check(d):
    assert d["schema"]=="polygon_v4"
    if d["noise_decision"]=="NO" or d["noise_confidence"]<TAU_NOISE:
        assert d["polygons"]==[]
        assert d["noise_type"]=="none"

done=set()
if OUT_POLY.exists():
    for l in OUT_POLY.open(encoding="utf-8"):
        r=json.loads(l); done.add((r["query_image_id"],r["candidate_image_id"]))

ok=fail=0
with IN_PAIRS.open(encoding="utf-8") as fin, \
     OUT_POLY.open("a",encoding="utf-8") as fout, \
     ERR_POLY.open("a",encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="v4 polygon"):
        row=json.loads(line)
        k=(row["query_image_id"],row["candidate_image_id"])
        if k in done: continue
        try:
            payload={
                "model":"qwen3-vl-8b",
                "messages":[
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":[
                        {"type":"text","text":USER},
                        {"type":"image_url","image_url":{"url":to_data_url(Path(row["query_path"]))}},
                    ]}
                ],
                "temperature":TEMP,
            }
            r=requests.post(LLM_URL,json=payload,timeout=240)
            r.raise_for_status()
            res=extract_json(r.json()["choices"][0]["message"]["content"])
            schema_check(res)

            out={
                "schema":"b2_polygon_allpairs_v4",
                "query_image_id":row["query_image_id"],
                "candidate_image_id":row["candidate_image_id"],
                "noisy_path":row["query_path"],
                **res,
                "tau_noise":TAU_NOISE,
            }
            fout.write(json.dumps(out,ensure_ascii=False)+"\n")
            ok+=1
        except Exception as e:
            ferr.write(json.dumps({"error":str(e),**row},ensure_ascii=False)+"\n")
            fail+=1

print("[DONE] v4 polygon", ok, fail)

# %% [cell 31]
# ===== Cell 2: v4 Visualization + NO audit =====
import json
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
POLY = CKPT / "b2_polygon_allpairs_v4.jsonl"

VIZ_ALL = ROOT / "viz_v4" / "pairs"
VIZ_NO  = ROOT / "viz_v4" / "no_noise_pairs"
VIZ_ALL.mkdir(parents=True,exist_ok=True)
VIZ_NO.mkdir(parents=True,exist_ok=True)

rows=[json.loads(l) for l in POLY.open(encoding="utf-8")]

no_cnt=0
for r in tqdm(rows, desc="v4 viz"):
    qid=r["query_image_id"][-12:]
    cid=r["candidate_image_id"][-12:]
    d=VIZ_ALL/f"{qid}__{cid}"
    d.mkdir(exist_ok=True)

    img=Image.open(r["noisy_path"]).convert("RGB")
    draw=ImageDraw.Draw(img,"RGBA")
    for p in r["polygons"]:
        pts=[(x*img.width,y*img.height) for x,y in p["points"]]
        draw.polygon(pts,outline=(255,0,0),fill=(255,0,0,80))
    img.save(d/"query_overlay.png")

    meta_path=d/"meta.json"
    meta_path.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")

    if r["noise_decision"]=="NO":
        no_cnt+=1
        d2=VIZ_NO/f"{qid}__{cid}"
        d2.mkdir(exist_ok=True)
        Image.open(r["noisy_path"]).save(d2/"noisy.png")
        Image.open(
            Path(r["noisy_path"]).with_name(
                Path(r["noisy_path"]).name.replace("303w","303u")
            )
        ).save(d2/"clean.png")
        (d2/"meta.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")

print("[NO COUNT]", no_cnt)
print("[VIZ]", VIZ_ALL)
print("[VIZ NO]", VIZ_NO)

# %% [cell 32]
# ===== Cell 3: v4 Mask-CLIP (NO -> no mask) =====
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --------- paths ---------
ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"

PAIRS_FILE   = CKPT / "pairs_303w_v2.jsonl"
POLY_FILE    = CKPT / "b2_polygon_allpairs_v4.jsonl"

OUT_FILE = CKPT / "c_maskclip_303w_v4.jsonl"
ERR_FILE = CKPT / "c_maskclip_303w_v4_errors.jsonl"

# --------- device ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[DEVICE]", device)

# --------- load CLIP (local) ---------
CLIP_PATH = r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14"
model = CLIPModel.from_pretrained(CLIP_PATH).to(device).eval()
processor = CLIPProcessor.from_pretrained(CLIP_PATH)
print("[OK] CLIP loaded")

# --------- resume ---------
done = set()
if OUT_FILE.exists():
    for line in OUT_FILE.open("r", encoding="utf-8"):
        try:
            r = json.loads(line)
            done.add((r["query_image_id"], r["candidate_image_id"]))
        except Exception:
            pass
print("[INFO] already done:", len(done))

# --------- helpers ---------
def load_img(p):
    return Image.open(p).convert("RGB")

def mask_image(img, polygons):
    """mask polygon regions to black"""
    if not polygons:
        return img
    w, h = img.size
    arr = np.array(img).copy()
    for p in polygons:
        pts = [(int(x*w), int(y*h)) for x,y in p["points"]]
        import cv2
        cv2.fillPoly(arr, [np.array(pts, dtype=np.int32)], (0,0,0))
    return Image.fromarray(arr)

def clip_sim(img1, img2):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0] * feats[1]).sum().item())

# --------- load inputs ---------
pairs = {(r["query_image_id"], r["candidate_image_id"]): r
         for r in (json.loads(l) for l in PAIRS_FILE.open(encoding="utf-8"))}

polys = {(r["query_image_id"], r["candidate_image_id"]): r
         for r in (json.loads(l) for l in POLY_FILE.open(encoding="utf-8"))}

# --------- main loop ---------
ok = fail = 0

with OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for key, row in tqdm(pairs.items(), desc="Mask-CLIP v4"):
        if key in done:
            continue
        try:
            qid, cid = key
            poly = polys[key]

            q_path = Path(row["query_path"])
            c_path = Path(row["candidate_path"])

            img_q = load_img(q_path)
            img_c = load_img(c_path)

            sim_raw = clip_sim(img_q, img_c)

            # ---- v4 核心规则 ----
            if poly["noise_decision"] == "NO":
                sim_mask = sim_raw
                delta = 0.0
                explain = "noise_decision=NO, no masking applied"
            else:
                img_q_mask = mask_image(img_q, poly["polygons"])
                sim_mask = clip_sim(img_q_mask, img_c)
                delta = sim_mask - sim_raw
                explain = f"masked {len(poly['polygons'])} polygons"

            out = {
                "schema": "c_maskclip_303w_v4",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "clip_raw": sim_raw,
                "clip_masked": sim_mask,
                "delta": delta,
                "explain": explain,
                "noise_decision": poly["noise_decision"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "c_maskclip_303w_v4_error",
                "query_image_id": key[0],
                "candidate_image_id": key[1],
                "error": str(e)
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            fail += 1

print("[DONE] Cell3 v4 mask-clip finished")
print("[STATS] ok:", ok, "fail:", fail)
print("[OUT]", OUT_FILE)
print("[ERR]", ERR_FILE)

# %% [cell 33]
# ===== Cell 4: summary + compare v1-v4 =====
import json, pandas as pd
from pathlib import Path

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

VERSIONS = ["v1","v2","v3","v3_1","v4"]
rows=[]

for v in VERSIONS:
    poly = CKPT / f"b2_polygon_allpairs_{v}.jsonl"
    if not poly.exists(): continue
    data=[json.loads(l) for l in poly.open(encoding="utf-8")]
    rows.append({
        "version":v,
        "n_pairs":len(data),
        "share_NO":sum(r.get("noise_decision")=="NO" for r in data)/len(data) if "noise_decision" in data[0] else None,
        "avg_n_poly":sum(len(r.get("polygons",[])) for r in data)/len(data),
    })

df=pd.DataFrame(rows)
df.to_csv(OUT/"table_v1_v4_overview.csv",index=False,encoding="utf-8-sig")
print(df)

# %% [cell 34]
# ===== Cell 4 (REVISED): v1/v2/v3/v3.1/v4 full compare (same as v3.1 table) =====
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"<PROTOTYPE303_ROOT>")
CKPT = ROOT / "checkpoints"
OUTS = ROOT / "outputs"
OUTS.mkdir(parents=True, exist_ok=True)

# ---- files (adjust names if yours differ) ----
FILES = {
    "v1": {
        "poly": CKPT / "b2_polygon_allpairs_v1.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v1.jsonl",
    },
    "v2": {
        "poly": CKPT / "b2_polygon_allpairs_v2.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v2.jsonl",
    },
    "v3": {
        "poly": CKPT / "b2_polygon_allpairs_v3.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v3.jsonl",
    },
    "v3_1": {
        "poly": CKPT / "b2_polygon_allpairs_v3_1.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v3_1.jsonl",
    },
    "v4": {
        "poly": CKPT / "b2_polygon_allpairs_v4.jsonl",
        "mc":   CKPT / "c_maskclip_303w_v4.jsonl",
    },
}

# ---------------- helpers ----------------
def load_jsonl(p: Path):
    assert p.exists(), f"Missing: {p}"
    out=[]
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            out.append(json.loads(line))
    return out

def poly_area_frac(polys):
    """
    Approx area fraction using polygon shoelace in normalized [0,1] coords.
    Multiple polygons: sum areas (ignores overlaps, acceptable for summary proxy).
    """
    if not polys:
        return 0.0
    total=0.0
    for p in polys:
        pts = p.get("points") or []
        if len(pts) < 3:
            continue
        xs=[float(x) for x,_ in pts]
        ys=[float(y) for _,y in pts]
        # shoelace
        s=0.0
        n=len(pts)
        for i in range(n):
            x1,y1 = float(pts[i][0]), float(pts[i][1])
            x2,y2 = float(pts[(i+1)%n][0]), float(pts[(i+1)%n][1])
            s += x1*y2 - x2*y1
        total += abs(s) * 0.5
    # clamp to [0,1]
    if total < 0: total = 0.0
    if total > 1: total = 1.0
    return float(total)

def summarize_polygon(records):
    n=len(records)
    noise_type_dist={}
    n_poly_dist={}
    cover_fracs=[]

    for r in records:
        # accept both schemas:
        # - older: {"polygons":[...], "noise_type":..., ...}
        # - v4: {"polygons":[...], "noise_decision":..., "noise_type":...}
        polys = r.get("polygons", []) or []
        nt = r.get("noise_type", None)
        noise_type_dist[str(nt)] = noise_type_dist.get(str(nt), 0) + 1

        k=str(len(polys))
        n_poly_dist[k] = n_poly_dist.get(k, 0) + 1
        cover_fracs.append(poly_area_frac(polys))

    cover_fracs=np.array(cover_fracs, dtype=float) if cover_fracs else np.array([], dtype=float)

    # share_empty_polygons: fraction with 0 polygons
    share_empty = float((cover_fracs==0).mean()) if n>0 else None

    def q(x):
        if x.size==0: 
            return None
        return float(np.quantile(x, 0.90))

    return {
        "n": n,
        "noise_type_dist": noise_type_dist,
        "n_poly_dist": n_poly_dist,
        "cover_frac_desc": {
            "mean": float(cover_fracs.mean()) if n>0 else None,
            "p50": float(np.quantile(cover_fracs, 0.50)) if n>0 else None,
            "p90": float(np.quantile(cover_fracs, 0.90)) if n>0 else None,
            "p95": float(np.quantile(cover_fracs, 0.95)) if n>0 else None,
            "max": float(cover_fracs.max()) if n>0 else None,
        },
        "share_empty_polygons": share_empty,
    }

def summarize_maskclip(records):
    deltas=[]
    for r in records:
        # accept both: delta field exists
        d = r.get("delta", None)
        if d is None:
            # fallback: clip_masked - clip_raw
            if "clip_masked" in r and "clip_raw" in r:
                d = float(r["clip_masked"]) - float(r["clip_raw"])
            else:
                continue
        deltas.append(float(d))
    deltas=np.array(deltas, dtype=float)
    n=int(deltas.size)

    def quant(q):
        return float(np.quantile(deltas, q)) if n>0 else None

    return {
        "n": n,
        "mean": float(deltas.mean()) if n>0 else None,
        "std": float(deltas.std(ddof=0)) if n>0 else None,
        "p01": quant(0.01),
        "p05": quant(0.05),
        "p50": quant(0.50),
        "p95": quant(0.95),
        "p99": quant(0.99),
        "min": float(deltas.min()) if n>0 else None,
        "max": float(deltas.max()) if n>0 else None,
        "abs_tail_counts": {
            "-0.1": int((deltas <= -0.1).sum()) if n>0 else None,
            "-0.2": int((deltas <= -0.2).sum()) if n>0 else None,
            "-0.3": int((deltas <= -0.3).sum()) if n>0 else None,
        }
    }

def share_NO(records):
    # only meaningful for v4, but we compute if field exists
    vals=[]
    for r in records:
        if "noise_decision" in r:
            vals.append(1 if r["noise_decision"]=="NO" else 0)
    if not vals:
        return None
    return float(np.mean(vals))

# ---------------- main ----------------
summary={}
table_rows=[]
poly_rows=[]
mc_rows=[]

for v, fp in FILES.items():
    poly_recs = load_jsonl(fp["poly"])
    mc_recs   = load_jsonl(fp["mc"])

    poly_sum = summarize_polygon(poly_recs)
    mc_sum   = summarize_maskclip(mc_recs)

    summary[v] = {
        "polygon": poly_sum,
        "maskclip": mc_sum,
        "share_NO": share_NO(poly_recs),
    }

    # ---- table_3criteria (match your screenshot) ----
    table_rows.append({
        "version": v,
        "Recall@K (303 stage)": "N/A (fixed candidate set)",
        "Dispute proxy: share_empty_polygons": poly_sum["share_empty_polygons"],
        "Dispute proxy: cover_frac_p90": poly_sum["cover_frac_desc"]["p90"],
        "Boundary: delta_std": mc_sum["std"],
        "Boundary: tail<= -0.1": mc_sum["abs_tail_counts"]["-0.1"],
        "Boundary: tail<= -0.2": mc_sum["abs_tail_counts"]["-0.2"],
        "Boundary: tail<= -0.3": mc_sum["abs_tail_counts"]["-0.3"],
        "Boundary: delta_mean": mc_sum["mean"],
        "Aux: share_NO (v4 only)": summary[v]["share_NO"],
    })

    # ---- richer per-version CSVs (optional but useful) ----
    poly_rows.append({
        "version": v,
        "n": poly_sum["n"],
        "share_empty_polygons": poly_sum["share_empty_polygons"],
        "cover_mean": poly_sum["cover_frac_desc"]["mean"],
        "cover_p50": poly_sum["cover_frac_desc"]["p50"],
        "cover_p90": poly_sum["cover_frac_desc"]["p90"],
        "cover_p95": poly_sum["cover_frac_desc"]["p95"],
        "cover_max": poly_sum["cover_frac_desc"]["max"],
        "share_NO": summary[v]["share_NO"],
        "noise_type_dist": json.dumps(poly_sum["noise_type_dist"], ensure_ascii=False),
        "n_poly_dist": json.dumps(poly_sum["n_poly_dist"], ensure_ascii=False),
    })

    mc_rows.append({
        "version": v,
        "n": mc_sum["n"],
        "delta_mean": mc_sum["mean"],
        "delta_std": mc_sum["std"],
        "p01": mc_sum["p01"],
        "p05": mc_sum["p05"],
        "p50": mc_sum["p50"],
        "p95": mc_sum["p95"],
        "p99": mc_sum["p99"],
        "min": mc_sum["min"],
        "max": mc_sum["max"],
        "tail<=-0.1": mc_sum["abs_tail_counts"]["-0.1"],
        "tail<=-0.2": mc_sum["abs_tail_counts"]["-0.2"],
        "tail<=-0.3": mc_sum["abs_tail_counts"]["-0.3"],
    })

# ---------------- write outputs ----------------
# 1) the same “3criteria” table style
df3 = pd.DataFrame(table_rows)
df3.to_csv(OUTS / "table_3criteria_v1_v2_v3_v3_1_v4.csv", index=False, encoding="utf-8-sig")

# 2) richer compare tables
pd.DataFrame(poly_rows).to_csv(OUTS / "compare_polygon_stats_v1_v2_v3_v3_1_v4.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(mc_rows).to_csv(OUTS / "compare_maskclip_stats_v1_v2_v3_v3_1_v4.csv", index=False, encoding="utf-8-sig")

# 3) summary json
(OUTS / "summary_compare_v1_v2_v3_v3_1_v4.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("[DONE] exported:")
print("[OUT]", OUTS / "table_3criteria_v1_v2_v3_v3_1_v4.csv")
print("[OUT]", OUTS / "compare_polygon_stats_v1_v2_v3_v3_1_v4.csv")
print("[OUT]", OUTS / "compare_maskclip_stats_v1_v2_v3_v3_1_v4.csv")
print("[OUT]", OUTS / "summary_compare_v1_v2_v3_v3_1_v4.json")

print("\n[PREVIEW table_3criteria]")
display(df3)

# %% [cell 35]
import json
import numpy as np
from pathlib import Path

# -------- paths --------
POLY_V4 = Path(r"<PROTOTYPE303_ROOT>\checkpoints\b2_polygon_v4.jsonl")
OUT_JSON = Path(r"<PROTOTYPE303_ROOT>\outputs\summary_v4_capped.json")
CAP = 0.2  # 关键：cap 阈值

# -------- load cover_frac --------
cover_fracs = []

with POLY_V4.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        # 你现在的结构里一般是 polygon_stats / cover_frac
        cf = r.get("polygon_stats", {}).get("cover_frac", None)
        if cf is not None:
            cover_fracs.append(cf)

cover_fracs = np.array(cover_fracs, dtype=float)

# -------- raw stats --------
raw_stats = {
    "n": int(len(cover_fracs)),
    "mean": float(np.mean(cover_fracs)),
    "p50": float(np.percentile(cover_fracs, 50)),
    "p90": float(np.percentile(cover_fracs, 90)),
    "p95": float(np.percentile(cover_fracs, 95)),
    "max": float(np.max(cover_fracs)),
}

# -------- capped stats --------
cover_fracs_cap = np.minimum(cover_fracs, CAP)

cap_stats = {
    "cap_value": CAP,
    "mean_cap": float(np.mean(cover_fracs_cap)),
    "p50_cap": float(np.percentile(cover_fracs_cap, 50)),
    "p90_cap": float(np.percentile(cover_fracs_cap, 90)),
    "p95_cap": float(np.percentile(cover_fracs_cap, 95)),
    "share_hit_cap": float(np.mean(cover_fracs > CAP)),
}

summary = {
    "v4_cover_frac_raw": raw_stats,
    "v4_cover_frac_capped": cap_stats,
}

OUT_JSON.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("[DONE] v4 capped summary written")
print(json.dumps(summary, ensure_ascii=False, indent=2))

# %% [cell 36]
import json
import numpy as np
from pathlib import Path

# -------- paths（与你截图完全一致） --------
POLY_V4 = Path(r"<PROTOTYPE303_ROOT>\checkpoints\b2_polygon_allpairs_v4.jsonl")
OUT_JSON = Path(r"<PROTOTYPE303_ROOT>\outputs\summary_v4_capped.json")

CAP = 0.2  # capped 阈值，可在论文中说明

assert POLY_V4.exists(), f"Missing polygon file: {POLY_V4}"

# -------- load cover_frac --------
cover_fracs = []

with POLY_V4.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        stats = r.get("polygon_stats", {})
        cf = stats.get("cover_frac", None)
        if cf is not None:
            cover_fracs.append(cf)

cover_fracs = np.asarray(cover_fracs, dtype=float)

print("[INFO] loaded cover_frac:", len(cover_fracs))

# -------- raw stats --------
raw_stats = {
    "n": int(len(cover_fracs)),
    "mean": float(np.mean(cover_fracs)),
    "p50": float(np.percentile(cover_fracs, 50)),
    "p90": float(np.percentile(cover_fracs, 90)),
    "p95": float(np.percentile(cover_fracs, 95)),
    "max": float(np.max(cover_fracs)),
}

# -------- capped stats --------
cover_cap = np.minimum(cover_fracs, CAP)

cap_stats = {
    "cap_value": CAP,
    "mean_cap": float(np.mean(cover_cap)),
    "p50_cap": float(np.percentile(cover_cap, 50)),
    "p90_cap": float(np.percentile(cover_cap, 90)),
    "p95_cap": float(np.percentile(cover_cap, 95)),
    "share_hit_cap": float(np.mean(cover_fracs > CAP)),
}

summary = {
    "version": "v4",
    "polygon_cover_frac_raw": raw_stats,
    "polygon_cover_frac_capped": cap_stats,
}

OUT_JSON.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("[DONE] v4 capped summary written to:", OUT_JSON)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# %% [cell 37]
import json
import numpy as np
from pathlib import Path

POLY_V4 = Path(r"<PROTOTYPE303_ROOT>\checkpoints\b2_polygon_allpairs_v4.jsonl")
OUT_JSON = Path(r"<PROTOTYPE303_ROOT>\outputs\summary_v4_coverfrac_capped.json")

CAP = 0.2  # 你也可以改成 0.1 / 0.3 做敏感性分析

assert POLY_V4.exists(), f"Missing: {POLY_V4}"

def poly_area_norm01(points):
    """
    points: [[x,y], ...] normalized to [0,1]
    returns polygon area in [0,1] (fraction of image) if points are valid
    """
    if not points or len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i+1) % n][0]), float(points[(i+1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def cover_frac_from_record(r):
    # v4: polygons is list of {"points": [[x,y], ...], ...}
    polys = r.get("polygons") or []
    if not polys:
        return 0.0
    tot = 0.0
    for p in polys:
        pts = p.get("points") or []
        tot += poly_area_norm01(pts)
    # 忽略 overlap，做个 clamp
    return float(min(max(tot, 0.0), 1.0))

cover_fracs = []
n_yes = 0
n_no = 0
n_lines = 0

with POLY_V4.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        n_lines += 1
        r = json.loads(line)
        if r.get("noise_decision") == "YES":
            n_yes += 1
        elif r.get("noise_decision") == "NO":
            n_no += 1
        cover_fracs.append(cover_frac_from_record(r))

cover_fracs = np.asarray(cover_fracs, dtype=float)

raw = {
    "n": int(len(cover_fracs)),
    "share_NO": float(n_no / max(n_lines, 1)),
    "share_YES": float(n_yes / max(n_lines, 1)),
    "mean": float(np.mean(cover_fracs)),
    "p50": float(np.percentile(cover_fracs, 50)),
    "p90": float(np.percentile(cover_fracs, 90)),
    "p95": float(np.percentile(cover_fracs, 95)),
    "max": float(np.max(cover_fracs)),
}

cap = np.minimum(cover_fracs, CAP)
capped = {
    "cap_value": CAP,
    "mean_cap": float(np.mean(cap)),
    "p50_cap": float(np.percentile(cap, 50)),
    "p90_cap": float(np.percentile(cap, 90)),
    "p95_cap": float(np.percentile(cap, 95)),
    "share_hit_cap": float(np.mean(cover_fracs > CAP)),
}

out = {"version": "v4", "cover_frac_raw": raw, "cover_frac_capped": capped}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

print("[DONE] v4 cover_frac+capped summary")
print("[OUT]", OUT_JSON)
print(json.dumps(out, ensure_ascii=False, indent=2))

# %% [cell 38]
import json
import pandas as pd
from pathlib import Path

# -------- inputs --------
IN_TABLE = Path(r"<PROTOTYPE303_ROOT>\outputs\table_3criteria_v1_v2_v3_v3_1_v4.csv")
V4_CAPPED_JSON = Path(r"<PROTOTYPE303_ROOT>\outputs\summary_v4_coverfrac_capped.json")

# -------- outputs --------
OUT_TABLE = Path(r"<PROTOTYPE303_ROOT>\outputs\table_3criteria_v1_v2_v3_v3_1_v4_withcap.csv")

assert IN_TABLE.exists(), f"Missing IN_TABLE: {IN_TABLE}"
assert V4_CAPPED_JSON.exists(), f"Missing V4_CAPPED_JSON: {V4_CAPPED_JSON}"

# -------- load base table --------
df = pd.read_csv(IN_TABLE, encoding="utf-8")

# -------- load v4 capped summary --------
s = json.loads(V4_CAPPED_JSON.read_text(encoding="utf-8"))

# v4 capped columns we want
p90_cap = float(s["cover_frac_capped"]["p90_cap"])
share_hit_cap = float(s["cover_frac_capped"]["share_hit_cap"])
cap_val = float(s["cover_frac_capped"]["cap_value"])

col_p90_cap = f"Dispute proxy: cover_frac_p90_cap{cap_val}"
col_share_hit = f"Dispute proxy: share_hit_cap{cap_val}"

# ensure columns exist
if col_p90_cap not in df.columns:
    df[col_p90_cap] = ""
if col_share_hit not in df.columns:
    df[col_share_hit] = ""

# write values only for v4 row
mask_v4 = df["version"].astype(str).str.lower().eq("v4")
assert mask_v4.any(), "No v4 row found in the input table (column 'version')."

df.loc[mask_v4, col_p90_cap] = p90_cap
df.loc[mask_v4, col_share_hit] = share_hit_cap

# keep numeric types where possible
df[col_p90_cap] = pd.to_numeric(df[col_p90_cap], errors="coerce")
df[col_share_hit] = pd.to_numeric(df[col_share_hit], errors="coerce")

# save
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

print("[DONE] merged v4 capped columns into table")
print("[IN ]", IN_TABLE)
print("[OUT]", OUT_TABLE)
print("\n[PREVIEW]")
print(df)

# %% [cell 39]

