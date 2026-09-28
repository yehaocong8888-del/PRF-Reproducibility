# AUTO-EXPORTED FROM ORIGINAL JUPYTER NOTEBOOK
# This file is DERIVED and was not executed during export.
# Source notebook: PRF1.ipynb

# %% [cell 1]
# ===== Cell 1: Global Config =====
import os
from pathlib import Path

PRF_ROOT = Path(r"<PRF_ROOT>")
JG_ROOT  = PRF_ROOT / "jg"

# 数据
ARCHIVE_ROOT = PRF_ROOT / "archive" / "303"
PATH_W = ARCHIVE_ROOT / "303w"   # watermarked (query)
PATH_U = ARCHIVE_ROOT / "303u"   # unwatermarked (gallery)
PATH_WJ = ARCHIVE_ROOT / "303wj" # gold bbox (sanity only)

# 输出
OUT_CKPT = JG_ROOT / "checkpoints"
OUT_LOG  = JG_ROOT / "logs"
OUT_TAB  = JG_ROOT / "tables"
OUT_CASE = JG_ROOT / "cases"
OUT_VIZ  = JG_ROOT / "viz"

for p in [OUT_CKPT, OUT_LOG, OUT_TAB, OUT_CASE, OUT_VIZ]:
    p.mkdir(parents=True, exist_ok=True)

# LLM endpoints
LLM_8B  = "http://127.0.0.1:8080/v1/chat/completions"
LLM_30B = "http://127.0.0.1:8081/v1/chat/completions"

TOPK_CLIP = 5
TOPK_8B   = 5
FINAL_TOPK = 10

# %% [cell 2]
# ===== Cell 2: Image Identity Utils =====
import hashlib
import numpy as np
from PIL import Image

def image_sha256(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()

def load_image(path: Path):
    return Image.open(path).convert("RGB")

# %% [cell 3]
# ===== Cell 3: LLM Client =====
import requests
import base64
import io

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def llm_chat(endpoint, messages, max_tokens=1024):
    payload = {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
    }
    r = requests.post(endpoint, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

def check_server(endpoint):
    health = endpoint.replace("/v1/chat/completions", "/health")
    r = requests.get(health, timeout=10)
    return r.json()

print("8B:", check_server(LLM_8B))
print("30B:", check_server(LLM_30B))

# %% [cell 4]
!pip install -U "huggingface-hub==0.30.1" "transformers==4.55.0" "safetensors>=0.4.3"

# %% [cell 5]
!pip uninstall -y huggingface-hub transformers tokenizers safetensors

# %% [cell 6]
!pip install -U "huggingface-hub==0.35.0" "transformers==4.55.0" "tokenizers==0.20.3" "safetensors>=0.4.3"

# %% [cell 7]
!pip install -U "huggingface-hub==0.35.0" "transformers==4.55.0" "safetensors>=0.4.3"

# %% [cell 8]
import sys
print("KERNEL PYTHON =", sys.executable)

# 先在当前 kernel 环境里卸载冲突版本
!"{sys.executable}" -m pip uninstall -y huggingface-hub transformers tokenizers safetensors

# 再在当前 kernel 环境里安装兼容组合
!"{sys.executable}" -m pip install -U "huggingface-hub==0.35.0" "transformers==4.55.0" "safetensors>=0.4.3"

# %% [cell 9]
import sys
print("KERNEL PYTHON =", sys.executable)

# 先在当前 kernel 环境里卸载冲突版本
!"{sys.executable}" -m pip uninstall -y huggingface-hub transformers tokenizers safetensors

# 再在当前 kernel 环境里安装兼容组合
!"{sys.executable}" -m pip install -U "huggingface-hub==0.35.0" "transformers==4.55.0" "safetensors>=0.4.3"

# %% [cell 10]
import transformers, huggingface_hub, tokenizers, safetensors
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("tokenizers:", tokenizers.__version__)
print("safetensors:", safetensors.__version__)

# %% [cell 11]
# ===== Cell 4 (FINAL): Load local HF CLIP (offline) =====
from pathlib import Path
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

LOCAL_CLIP_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\cache\hf\openai\clip-vit-large-patch14")
assert LOCAL_CLIP_DIR.exists(), f"Local CLIP dir not found: {LOCAL_CLIP_DIR}"

clip_model = CLIPModel.from_pretrained(
    str(LOCAL_CLIP_DIR),
    local_files_only=True
).to(device).eval()

clip_processor = CLIPProcessor.from_pretrained(
    str(LOCAL_CLIP_DIR),
    local_files_only=True
)

print("[OK] Local CLIP loaded from:", LOCAL_CLIP_DIR)

# %% [cell 12]
# ===== Cell 5 (FINAL): Embed cache (resume-safe, no filename matching) =====
import json, hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# 输出路径（全部在 PRF/jg）
CKPT_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_EMB_JSONL = CKPT_DIR / "clip_hf_vitl14_emb.jsonl"  # 事实源：逐条落盘

def image_sha256_pixels(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()

@torch.no_grad()
def clip_embed_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = clip_model.get_image_features(**inputs)  # [1, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().float().cpu().numpy()[0]

# 断点恢复：加载已算 embedding
emb_cache = {}   # image_id -> embedding(list[float])
path_cache = {}  # image_id -> path(str)

if CKPT_EMB_JSONL.exists():
    with CKPT_EMB_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            emb_cache[j["image_id"]] = j["embedding"]
            path_cache[j["image_id"]] = j["path"]

print(f"[OK] loaded embeddings: {len(emb_cache)}")

def ensure_clip_embedding(path: Path):
    """
    Returns: (image_id, embedding_list)
    - image_id is sha256 of pixel bytes, not filename
    - embedding is L2-normalized
    - first time computes then appends to JSONL immediately
    """
    img_id = "sha256:" + image_sha256_pixels(path)
    if img_id in emb_cache:
        return img_id, emb_cache[img_id]

    emb = clip_embed_image(path).tolist()
    rec = {
        "schema": "clip_hf_emb_v1",
        "model": "openai/clip-vit-large-patch14",
        "image_id": img_id,
        "path": str(path),
        "display_name": path.name,  # 仅用于显示/追溯，不参与任何匹配
        "embedding": emb,
    }
    with CKPT_EMB_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    emb_cache[img_id] = emb
    path_cache[img_id] = str(path)
    return img_id, emb

# %% [cell 13]
# ===== Cell 6 (FINAL): Build embeddings for 303u / 303w (resume-safe) =====
from pathlib import Path

PATH_U = Path(r"<PRF_ROOT>\archive\303\303u")
PATH_W = Path(r"<PRF_ROOT>\archive\303\303w")

u_imgs = sorted([p for p in PATH_U.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
w_imgs = sorted([p for p in PATH_W.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])

print("303u images:", len(u_imgs))
print("303w images:", len(w_imgs))

# 建议顺序：先 gallery 再 query
for p in tqdm(u_imgs, desc="embed 303u"):
    ensure_clip_embedding(p)

for p in tqdm(w_imgs, desc="embed 303w"):
    ensure_clip_embedding(p)

print("[DONE] total embeddings:", len(emb_cache))
print("[OUT] jsonl:", CKPT_EMB_JSONL)

# %% [cell 14]
# ===== Cell 7: Build FAISS index for 303u gallery =====
import json
import numpy as np
import faiss
from pathlib import Path

CKPT_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
EMB_JSONL = CKPT_DIR / "clip_hf_vitl14_emb.jsonl"
FAISS_INDEX_PATH = CKPT_DIR / "faiss_303u.index"
FAISS_META_PATH  = CKPT_DIR / "faiss_303u_meta.json"

# 读取 embedding，只取 303u
gallery_ids = []
gallery_embs = []

with EMB_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        if "\\303u\\" in j["path"]:
            gallery_ids.append(j["image_id"])
            gallery_embs.append(j["embedding"])

gallery_embs = np.asarray(gallery_embs, dtype="float32")
dim = gallery_embs.shape[1]

print("Gallery size:", len(gallery_ids), "dim:", dim)

# 使用 Inner Product（因为我们已做 L2 normalize）
index = faiss.IndexFlatIP(dim)
index.add(gallery_embs)

faiss.write_index(index, str(FAISS_INDEX_PATH))

with FAISS_META_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "schema": "faiss_gallery_v1",
            "model": "openai/clip-vit-large-patch14",
            "count": len(gallery_ids),
            "image_ids": gallery_ids,
        },
        f,
        indent=2
    )

print("[OK] FAISS index saved:", FAISS_INDEX_PATH)

# %% [cell 15]
# ===== Cell 8: CLIP Top-K retrieval (303w queries) =====
import json
import numpy as np
import faiss
from pathlib import Path

TOPK = 5

CKPT_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
EMB_JSONL = CKPT_DIR / "clip_hf_vitl14_emb.jsonl"
FAISS_INDEX_PATH = CKPT_DIR / "faiss_303u.index"
FAISS_META_PATH  = CKPT_DIR / "faiss_303u_meta.json"
TOPK_JSONL = CKPT_DIR / "clip_topk_303w.jsonl"

# load faiss
index = faiss.read_index(str(FAISS_INDEX_PATH))
meta = json.load(FAISS_META_PATH.open("r", encoding="utf-8"))
gallery_ids = meta["image_ids"]

# load all embeddings
emb_by_id = {}
path_by_id = {}

with EMB_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        emb_by_id[j["image_id"]] = np.asarray(j["embedding"], dtype="float32")
        path_by_id[j["image_id"]] = j["path"]

# run retrieval
with TOPK_JSONL.open("w", encoding="utf-8") as fw:
    for img_id, emb in emb_by_id.items():
        if "\\303w\\" not in path_by_id[img_id]:
            continue

        emb = emb.reshape(1, -1)
        scores, idxs = index.search(emb, TOPK)

        rec = {
            "schema": "clip_topk_v1",
            "query_image_id": img_id,
            "query_path": path_by_id[img_id],
            "topk": [
                {
                    "rank": int(r),
                    "gallery_image_id": gallery_ids[i],
                    "gallery_path": path_by_id[gallery_ids[i]],
                    "score": float(scores[0][r]),
                }
                for r, i in enumerate(idxs[0])
            ],
        }
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("[OK] CLIP Top-K saved:", TOPK_JSONL)

# %% [cell 16]
# ===== Cell 9: 8B Suggest Top-K' (semantic candidate expansion) =====
import json
import requests
from pathlib import Path

TOPK_SUGGEST = 5
LLM_8B_URL = "http://127.0.0.1:8080/v1/chat/completions"

CKPT_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
TOPK_JSONL = CKPT_DIR / "clip_topk_303w.jsonl"
SUGGEST_JSONL = CKPT_DIR / "8b_suggest_topk_303w.jsonl"

def call_8b_suggest(query_path, gallery_paths):
    prompt = f"""
You are reviewing image retrieval candidates.

Query image: {query_path}

Candidate gallery images:
{chr(10).join(f"- {p}" for p in gallery_paths)}

Task:
Suggest up to {TOPK_SUGGEST} additional gallery images (by index or description)
that may be semantically relevant but possibly missed by CLIP similarity.

Return JSON:
{{"suggested_indices":[...]}}
"""
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "temperature": 0.2,
    }
    r = requests.post(LLM_8B_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

with TOPK_JSONL.open("r", encoding="utf-8") as fr, \
     SUGGEST_JSONL.open("w", encoding="utf-8") as fw:

    for line in fr:
        j = json.loads(line)
        gallery_paths = [x["gallery_path"] for x in j["topk"]]

        try:
            rsp = call_8b_suggest(j["query_path"], gallery_paths)
            rec = {
                "schema": "8b_suggest_v1",
                "query_image_id": j["query_image_id"],
                "query_path": j["query_path"],
                "clip_topk": gallery_paths,
                "llm_raw": rsp,
            }
        except Exception as e:
            rec = {
                "schema": "8b_suggest_v1",
                "query_image_id": j["query_image_id"],
                "query_path": j["query_path"],
                "error": str(e),
            }

        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("[OK] 8B suggest saved:", SUGGEST_JSONL)

# %% [cell 17]
# ===== Cell 10: Build final candidate pool (union CLIP + 8B-suggest) =====
import json
from pathlib import Path

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")

CLIP_TOPK = CKPT / "clip_topk_303w.jsonl"
SUGGEST   = CKPT / "8b_suggest_topk_303w.jsonl"
UNION_OUT = CKPT / "final_candidates_303w.jsonl"

# load 8B suggest by query
suggest_map = {}
with SUGGEST.open("r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        if "llm_raw" in j:
            suggest_map[j["query_image_id"]] = j

with UNION_OUT.open("w", encoding="utf-8") as fw, \
     CLIP_TOPK.open("r", encoding="utf-8") as fr:

    for line in fr:
        j = json.loads(line)
        qid = j["query_image_id"]

        clip_ids = [x["gallery_image_id"] for x in j["topk"]]
        extra_ids = []

        if qid in suggest_map:
            # 简化处理：只记录 raw，具体解析后面再 refine
            extra_ids = []

        final_ids = list(dict.fromkeys(clip_ids + extra_ids))

        fw.write(json.dumps({
            "schema": "final_candidates_v1",
            "query_image_id": qid,
            "query_path": j["query_path"],
            "candidate_gallery_ids": final_ids,
            "clip_topk": clip_ids,
            "has_8b_suggest": qid in suggest_map
        }) + "\n")

print("[OK] Final candidate pool frozen:", UNION_OUT)

# %% [cell 18]
# ===== Cell 11: A/B/C judgments (no teacher) =====
import json
from pathlib import Path

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
CAND = CKPT / "final_candidates_303w.jsonl"
ABC_OUT = CKPT / "abc_scores_303w.jsonl"

with CAND.open("r", encoding="utf-8") as fr, \
     ABC_OUT.open("w", encoding="utf-8") as fw:

    for line in fr:
        j = json.loads(line)

        # A: CLIP score proxy（是否在 top-k）
        A = 1  

        # B: 8B review（此处先占位，后面接你真实 prompt）
        B = None  # TODO: call 8B review

        # C: Mask-CLIP（占位）
        C = None  # TODO: use 8B bbox + masked clip

        fw.write(json.dumps({
            "schema": "abc_score_v1",
            "query_image_id": j["query_image_id"],
            "candidates": j["candidate_gallery_ids"],
            "A": A,
            "B": B,
            "C": C
        }) + "\n")

print("[OK] A/B/C scores dumped:", ABC_OUT)

# %% [cell 19]
# ===== Cell 12: Dispute gating + optional 30B teacher =====
import json
from pathlib import Path

USE_TEACHER = True  # 消融点：False = 去掉 30B

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
ABC = CKPT / "abc_scores_303w.jsonl"
FINAL = CKPT / "final_labels_303w.jsonl"

with ABC.open("r", encoding="utf-8") as fr, \
     FINAL.open("w", encoding="utf-8") as fw:

    for line in fr:
        j = json.loads(line)

        dispute = False
        # 示例规则：B/C 不一致 或 置信不足
        if j["B"] is None or j["C"] is None:
            dispute = True

        final = None
        teacher_called = False

        if dispute and USE_TEACHER:
            teacher_called = True
            final = None  # TODO: call 30B
        else:
            final = j["A"]  # 简化：用 A 或 majority

        fw.write(json.dumps({
            "schema": "final_label_v1",
            "query_image_id": j["query_image_id"],
            "final": final,
            "dispute": dispute,
            "teacher_called": teacher_called
        }) + "\n")

print("[OK] Final labels saved:", FINAL)

# %% [cell 20]
# ===== Cell 13: Summarize ablations (303-only) =====
import pandas as pd
from pathlib import Path

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
FINAL = CKPT / "final_labels_303w.jsonl"

rows = []
with FINAL.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
out = CKPT / "ablation_summary_303.csv"
df.to_csv(out, index=False)

print("[OK] Ablation summary saved:", out)

# %% [cell 21]
# ===== Cell 14 (v2): 8B Pair Review with Chinese Rationales (303-only) =====

import json
import requests
from pathlib import Path
from tqdm import tqdm

# ---------------- paths ----------------
CKPT_DIR = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_FILE = CKPT_DIR / "pairs_303w.jsonl"   # 你之前 CLIP Top-K 的输出
OUT_FILE  = CKPT_DIR / "b1_8b_pair_review_303_v2.jsonl"

IMG_W_DIR = Path(r"<PRF_ROOT>\archive\303\303w")
IMG_U_DIR = Path(r"<PRF_ROOT>\archive\303\303u")

# ---------------- resume ----------------
done_pairs = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            done_pairs.add((j["query_image_id"], j["candidate_image_id"]))

print("[INFO] already reviewed:", len(done_pairs))

# ---------------- 8B endpoint ----------------
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "你是一个图像配对审核专家，负责判断两个图像是否值得作为训练配对。"
    "你的目标不是给分数，而是给出可解释、可审查的判断依据。"
)

USER_PROMPT = """
请你从以下五个维度分别判断，并对每一项给出：
- value：0 或 1
- comment：一句中文参考意见，说明你为什么这样判断

五个维度：
1. pixel_consistency（像素级一致性）
2. same_base_image（是否同一底图）
3. same_instance（是否同一物理实例）
4. strong_structure（强结构一致性）
5. weak_structure（弱结构一致性）

然后给出：
- final_recommendation：ACCEPT / REJECT / AMBIGUOUS
- confidence：0 到 1 之间的数值，表示你对最终判断的确信程度
- summary_comment：一句中文总结，概括你最终立场（供更大模型复核使用）

请【只用严格 JSON】输出，格式如下：

{
  "criteria": {
    "pixel_consistency": {"value": 0, "comment": "..."},
    "same_base_image": {"value": 0, "comment": "..."},
    "same_instance": {"value": 1, "comment": "..."},
    "strong_structure": {"value": 1, "comment": "..."},
    "weak_structure": {"value": 0, "comment": "..."}
  },
  "final_recommendation": "ACCEPT",
  "confidence": 0.82,
  "summary_comment": "..."
}
"""

def call_8b_review_v2(query_img: Path, cand_img: Path):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": query_img.as_uri()}},
                    {"type": "image_url", "image_url": {"url": cand_img.as_uri()}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(txt)

# ---------------- main loop ----------------
with TOPK_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="B1 8B pair review v2"):
        row = json.loads(line)

        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)

        if key in done_pairs:
            continue

        q_path = IMG_W_DIR / row["query_filename"]
        c_path = IMG_U_DIR / row["candidate_filename"]

        try:
            res = call_8b_review_v2(q_path, c_path)

            out = {
                "schema": "b1_pair_review_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": res["confidence"],
                "summary_comment": res["summary_comment"],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print("[WARN] skip pair:", qid, cid, e)

print("[OK] B1 v2 finished:", OUT_FILE)

# %% [cell 22]
import json
from pathlib import Path

p = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints\final_candidates_303w.jsonl")
with p.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        print(row.keys())
        if i >= 2:
            break

# %% [cell 23]
# ===== Cell 14A: Expand final_candidates into pair-level JSONL =====
import json
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
IN_FILE  = CKPT / "final_candidates_303w.jsonl"
OUT_FILE = CKPT / "pairs_303w.jsonl"

seen = set()
n_out = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="expand to pairs"):
        row = json.loads(line)

        qid   = row["query_image_id"]
        qpath = row["query_path"]

        for cid in row["candidate_gallery_ids"]:
            key = (qid, cid)
            if key in seen:
                continue
            seen.add(key)

            out = {
                "schema": "pair_v1",
                "query_image_id": qid,
                "query_path": qpath,
                "candidate_image_id": cid,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

print("[OK] pair file written:", OUT_FILE)
print("[OK] total pairs:", n_out)

# %% [cell 24]
# ===== Cell 14 (FINAL): Build pair-level mother table (303w) =====
import json
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")

IN_FILE  = CKPT / "final_candidates_303w.jsonl"
OUT_FILE = CKPT / "pairs_303w.jsonl"

assert IN_FILE.exists(), f"Missing input file: {IN_FILE}"

# ---------------- load existing pairs (resume-safe) ----------------
existing = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                existing.add((r["query_image_id"], r["candidate_image_id"]))
            except Exception:
                pass

print("[INFO] existing pairs:", len(existing))

# ---------------- expand ----------------
n_new = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="expand to pairs"):
        row = json.loads(line)

        # ---- strict schema check ----
        assert "query_image_id" in row
        assert "query_path" in row
        assert "candidate_gallery_ids" in row

        qid   = row["query_image_id"]
        qpath = row["query_path"]
        cids  = row["candidate_gallery_ids"]

        assert isinstance(cids, list) and len(cids) > 0

        for cid in cids:
            key = (qid, cid)
            if key in existing:
                continue

            out = {
                "schema": "pair_v1",
                "query_image_id": qid,
                "query_path": qpath,          # full path, no guessing
                "candidate_image_id": cid,    # pure ID, no filename coupling
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            existing.add(key)
            n_new += 1

print("[OK] pair file:", OUT_FILE)
print("[OK] total pairs:", len(existing))
print("[OK] newly added:", n_new)

# %% [cell 25]
# ===== Cell 15 (FINAL): B1 - 8B Pair Review v2 (303w, pair-level) =====
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ---------------- paths ----------------
CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w.jsonl"
OUT_FILE = CKPT / "b1_pair_review_303w_v2.jsonl"

assert IN_FILE.exists(), f"Missing input file: {IN_FILE}"

# ---------------- resume ----------------
done = set()
if OUT_FILE.exists():
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                done.add((j["query_image_id"], j["candidate_image_id"]))
            except Exception:
                pass

print("[INFO] already reviewed pairs:", len(done))

# ---------------- LLM endpoint ----------------
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "你是一个图像配对审核专家，负责判断两个图像是否值得作为训练配对。"
    "你的目标是给出可解释、可审查的判断依据，而不是打分排名。"
)

USER_PROMPT = """
请你对给定的两张图像进行配对审核，并严格按以下要求输出【JSON】：

对每一个维度，分别给出：
- value：0 或 1
- comment：一句【中文】参考意见，说明判断理由

五个维度：
1. pixel_consistency（像素级一致性）
2. same_base_image（是否同一底图）
3. same_instance（是否同一物理实例）
4. strong_structure（强结构一致性）
5. weak_structure（弱结构一致性）

然后给出：
- final_recommendation：ACCEPT / REJECT / AMBIGUOUS
- confidence：0 到 1 之间，表示你对最终判断的确信程度
- summary_comment：一句【中文】总结性说明，概括你最终立场

【重要】：
- 只输出 JSON
- 不要输出任何额外文本
- comment 与 summary_comment 必须是中文

JSON 格式示例（仅示意，不要照抄内容）：
{
  "criteria": {
    "pixel_consistency": {"value": 0, "comment": "..."},
    "same_base_image": {"value": 0, "comment": "..."},
    "same_instance": {"value": 1, "comment": "..."},
    "strong_structure": {"value": 1, "comment": "..."},
    "weak_structure": {"value": 0, "comment": "..."}
  },
  "final_recommendation": "ACCEPT",
  "confidence": 0.82,
  "summary_comment": "..."
}
"""

def call_8b_pair_review(query_path: Path, cand_path: Path):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": query_path.as_uri()}},
                    {"type": "image_url", "image_url": {"url": cand_path.as_uri()}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(txt)

# ---------------- main loop ----------------
n_new = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="B1 8B pair review v2"):
        row = json.loads(line)

        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)

        if key in done:
            continue

        q_path = Path(row["query_path"])

        # ⚠️ candidate 路径：假设 gallery 在 303u 下，后缀你可按实际改
        cand_path = Path(r"<PRF_ROOT>\archive\303\303u") / f"{cid}.png"

        try:
            res = call_8b_pair_review(q_path, cand_path)

            out = {
                "schema": "b1_pair_review_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": res["confidence"],
                "summary_comment": res["summary_comment"],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_new += 1

        except Exception as e:
            print("[WARN] skip pair:", qid, cid, e)

print("[OK] B1 v2 finished:", OUT_FILE)
print("[OK] newly added reviews:", n_new)

# %% [cell 26]
# ===== Cell 16 (FINAL): Dispute Gating + Teacher (30B) =====
import json
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")

PAIR_FILE   = CKPT / "pairs_303w.jsonl"
B1_FILE     = CKPT / "b1_pair_review_303w_v2.jsonl"
DISPUTE_OUT = CKPT / "dispute_303w.jsonl"
TEACHER_OUT = CKPT / "teacher_30b_303w.jsonl"

CONF_THR = 0.75

# ---------------- load B1 results ----------------
b1_map = {}
with B1_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["query_image_id"], r["candidate_image_id"])
        b1_map[key] = r

print("[INFO] loaded B1 reviews:", len(b1_map))

# ---------------- resume teacher ----------------
teacher_done = set()
if TEACHER_OUT.exists():
    with TEACHER_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            teacher_done.add((r["query_image_id"], r["candidate_image_id"]))

print("[INFO] teacher already done:", len(teacher_done))

# ---------------- Teacher endpoint ----------------
LLM_URL_30B = "http://127.0.0.1:8081/v1/chat/completions"

TEACHER_SYSTEM = (
    "你是最终裁判模型，负责在存在争议时，对图像配对是否值得作为训练样本做出裁决。"
)

def call_teacher(query_path: Path, cand_path: Path, b1_json: dict):
    prompt = (
        "下面是一个 8B 模型对该图像对的结构化评审结果（含五个判据及中文理由）。\n"
        "请你判断是否同意其最终结论。\n"
        "如果不同意，请明确指出你不同意的是哪一个判据以及原因。\n\n"
        "8B 评审结果：\n"
        f"{json.dumps(b1_json, ensure_ascii=False)}\n\n"
        "请只输出 JSON，包含：\n"
        "- final_recommendation: ACCEPT / REJECT / AMBIGUOUS\n"
        "- confidence: 0~1\n"
        "- explanation: 中文说明\n"
    )

    payload = {
        "model": "qwen3-vl-30b",
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": query_path.as_uri()}},
                    {"type": "image_url", "image_url": {"url": cand_path.as_uri()}},
                ],
            },
        ],
        "temperature": 0.0,
    }

    r = requests.post(LLM_URL_30B, json=payload, timeout=180)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(txt)

# ---------------- dispute logic ----------------
def is_dispute(b1):
    if b1["final_recommendation"] == "AMBIGUOUS":
        return True
    if b1["confidence"] < CONF_THR:
        return True

    c = b1["criteria"]
    # 逻辑矛盾示例（可扩展）
    if c["same_instance"]["value"] == 1 and \
       c["strong_structure"]["value"] == 0 and \
       c["weak_structure"]["value"] == 0:
        return True

    return False

# ---------------- main loop ----------------
n_dispute = 0
n_teacher = 0

with PAIR_FILE.open("r", encoding="utf-8") as fin, \
     DISPUTE_OUT.open("w", encoding="utf-8") as fd, \
     TEACHER_OUT.open("a", encoding="utf-8") as ft:

    for line in tqdm(fin, desc="dispute + teacher"):
        pair = json.loads(line)
        key = (pair["query_image_id"], pair["candidate_image_id"])

        b1 = b1_map.get(key)
        if b1 is None:
            continue

        dispute = is_dispute(b1)

        fd.write(json.dumps({
            "schema": "dispute_v1",
            "query_image_id": key[0],
            "candidate_image_id": key[1],
            "dispute": dispute,
            "confidence": b1["confidence"],
            "b1_final": b1["final_recommendation"]
        }, ensure_ascii=False) + "\n")

        if not dispute:
            continue

        n_dispute += 1

        if key in teacher_done:
            continue

        q_path = Path(pair["query_path"])
        c_path = Path(r"<PRF_ROOT>\archive\303\303u") / f"{key[1]}.png"

        try:
            tres = call_teacher(q_path, c_path, b1)
            out = {
                "schema": "teacher_30b_v1",
                "query_image_id": key[0],
                "candidate_image_id": key[1],
                **tres
            }
            ft.write(json.dumps(out, ensure_ascii=False) + "\n")
            ft.flush()
            teacher_done.add(key)
            n_teacher += 1

        except Exception as e:
            print("[WARN] teacher skip:", key, e)

print("[OK] total dispute pairs:", n_dispute)
print("[OK] teacher calls this run:", n_teacher)

# %% [cell 27]
from pathlib import Path

p = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints\b1_pair_review_303w_v2.jsonl")
print("exists:", p.exists())
if p.exists():
    print("size:", p.stat().st_size)

# %% [cell 28]
# ===== Cell 15 (DIAGNOSTIC + SAFE): B1 - 8B Pair Review v2 =====
import json
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w.jsonl"
OUT_FILE = CKPT / "b1_pair_review_303w_v2.jsonl"

IMG_U_DIR = Path(r"<PRF_ROOT>\archive\303\303u")

# -------- helper: resolve candidate image path --------
def resolve_cand_path(cid):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = IMG_U_DIR / f"{cid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate image not found for id={cid}")

# -------- resume --------
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

SYSTEM_PROMPT = (
    "你是一个图像配对审核专家，负责判断两个图像是否值得作为训练配对。"
)

USER_PROMPT = """
请你对给定的两张图像进行配对审核，并严格按以下要求输出【JSON】：

对每一个维度，分别给出：
- value：0 或 1
- comment：一句【中文】参考意见

五个维度：
1. pixel_consistency
2. same_base_image
3. same_instance
4. strong_structure
5. weak_structure

然后给出：
- final_recommendation：ACCEPT / REJECT / AMBIGUOUS
- confidence：0 到 1
- summary_comment：一句中文总结

【只输出 JSON，不要输出其他内容】
"""

def call_8b_pair_review(query_path: Path, cand_path: Path):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": query_path.as_uri()}},
                    {"type": "image_url", "image_url": {"url": cand_path.as_uri()}},
                ],
            },
        ],
        "temperature": 0.0,
    }

    r = requests.post(LLM_URL, json=payload, timeout=120)
    r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw)
    except Exception:
        print("[BAD JSON RAW OUTPUT]")
        print(raw)
        raise

# -------- main loop --------
n_ok = 0
n_fail = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="B1 diagnostic run"):
        row = json.loads(line)
        key = (row["query_image_id"], row["candidate_image_id"])
        if key in done:
            continue

        try:
            q_path = Path(row["query_path"])
            c_path = resolve_cand_path(row["candidate_image_id"])

            res = call_8b_pair_review(q_path, c_path)

            fout.write(json.dumps({
                "schema": "b1_pair_review_v2",
                "query_image_id": key[0],
                "candidate_image_id": key[1],
                **res
            }, ensure_ascii=False) + "\n")
            fout.flush()

            n_ok += 1

        except Exception as e:
            n_fail += 1
            print("[FAIL]", key, e)

print("[DONE] B1 diagnostic finished")
print("[STATS] success:", n_ok, "fail:", n_fail)

# %% [cell 29]
# ===== Cell 15 (FINAL v3): B1 - 8B Pair Review v2 with data URLs =====
import json
import time
import base64
import requests
from pathlib import Path
from tqdm import tqdm

# ---------------- paths ----------------
CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w.jsonl"
OUT_FILE = CKPT / "b1_pair_review_303w_v2.jsonl"
ERR_FILE = CKPT / "b1_pair_review_303w_v2_errors.jsonl"

IMG_U_DIR = Path(r"<PRF_ROOT>\archive\303\303u")  # gallery (clean)

assert IN_FILE.exists(), f"Missing input file: {IN_FILE}"

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
print("[INFO] out_file:", OUT_FILE)
print("[INFO] err_file:", ERR_FILE)

# ---------------- LLM endpoint ----------------
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "你是一个图像配对审核专家，负责判断两个图像是否值得作为训练配对。"
    "你需要输出可解释、可审查的结构化判据。"
)

USER_PROMPT = """
请你对给定的两张图像进行配对审核，并严格按以下要求输出【JSON】：

对每一个维度，分别给出：
- value：0 或 1
- comment：一句【中文】参考意见，说明判断理由

五个维度：
1. pixel_consistency（像素级一致性）
2. same_base_image（是否同一底图）
3. same_instance（是否同一物理实例）
4. strong_structure（强结构一致性）
5. weak_structure（弱结构一致性）

然后给出：
- final_recommendation：ACCEPT / REJECT / AMBIGUOUS
- confidence：0 到 1 之间
- summary_comment：一句【中文】总结性说明

【重要】：
- 只输出 JSON，不要输出任何额外文本
- comment 与 summary_comment 必须是中文
"""

# ---------------- helpers ----------------
def resolve_cand_path(cid: str) -> Path:
    # 自动探测后缀，避免 .png/.jpg 不一致导致全失败
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = IMG_U_DIR / f"{cid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate image not found for id={cid} under {IMG_U_DIR}")

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

def extract_json_from_text(txt: str) -> dict:
    # 有些模型会包一层 ```json ... ```，这里做稳健解析
    txt = txt.strip()
    if txt.startswith("```"):
        # 去掉代码块围栏
        txt = txt.strip("`")
        # 有时第一行是 json
        lines = txt.splitlines()
        if lines and lines[0].strip().lower() == "json":
            txt = "\n".join(lines[1:])
    # 尝试直接 loads
    return json.loads(txt)

def call_8b_pair_review(query_path: Path, cand_path: Path) -> dict:
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": to_data_url(query_path)}},
                    {"type": "image_url", "image_url": {"url": to_data_url(cand_path)}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    r = requests.post(LLM_URL, json=payload, timeout=240)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json_from_text(raw)

def schema_check(res: dict) -> None:
    # 强 schema 防止后续 N=0
    assert "criteria" in res and isinstance(res["criteria"], dict)
    for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        assert k in res["criteria"]
        assert "value" in res["criteria"][k]
        assert res["criteria"][k]["value"] in [0,1]
        assert "comment" in res["criteria"][k]
        assert isinstance(res["criteria"][k]["comment"], str)
    assert res["final_recommendation"] in ["ACCEPT","REJECT","AMBIGUOUS"]
    assert isinstance(res["confidence"], (int, float))
    assert 0.0 <= float(res["confidence"]) <= 1.0
    assert isinstance(res["summary_comment"], str)

# ---------------- main loop ----------------
n_ok = 0
n_fail = 0

with IN_FILE.open("r", encoding="utf-8") as fin, \
     OUT_FILE.open("a", encoding="utf-8") as fout, \
     ERR_FILE.open("a", encoding="utf-8") as ferr:

    for line in tqdm(fin, desc="B1 8B pair review v2 (data-url)"):
        row = json.loads(line)

        qid = row["query_image_id"]
        cid = row["candidate_image_id"]
        key = (qid, cid)

        if key in done:
            continue

        q_path = Path(row["query_path"])

        try:
            c_path = resolve_cand_path(cid)
            res = call_8b_pair_review(q_path, c_path)
            schema_check(res)

            out = {
                "schema": "b1_pair_review_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": float(res["confidence"]),
                "summary_comment": res["summary_comment"],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

        except Exception as e:
            n_fail += 1
            ferr.write(json.dumps({
                "schema": "b1_pair_review_v2_error",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "error": str(e),
                "query_path": str(q_path),
            }, ensure_ascii=False) + "\n")
            ferr.flush()

        # 可选：轻微节流，避免 server 短时过载
        # time.sleep(0.02)

print("[DONE] B1 v2 finished")
print("[STATS] success:", n_ok, "fail:", n_fail)
print("[OUT]", OUT_FILE)
print("[ERR]", ERR_FILE)

# %% [cell 30]
import json
from pathlib import Path

ERR_FILE = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints\b1_pair_review_303w_v2_errors.jsonl")
print("exists:", ERR_FILE.exists(), "size:", ERR_FILE.stat().st_size if ERR_FILE.exists() else None)

with ERR_FILE.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        e = json.loads(line)
        print("----", i, "----")
        print("qid:", e.get("query_image_id"), "cid:", e.get("candidate_image_id"))
        print("query_path:", e.get("query_path"))
        print("error:", e.get("error"))

# %% [cell 31]
import json
from collections import Counter
from pathlib import Path

ERR_FILE = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints\b1_pair_review_303w_v2_errors.jsonl")

c = Counter()
n = 0
with ERR_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        msg = e.get("error", "")
        # 只取前 120 字符做聚类（足够判断是哪一类错）
        c[msg[:120]] += 1
        n += 1

print("total errors:", n)
for k, v in c.most_common(20):
    print(v, " :: ", k)

# %% [cell 32]
# ===== Cell 14 (FIXED): build sha256->path map for 303u, then expand topk to pairs with candidate_path =====
import json, hashlib
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
TOPK_FILE = CKPT / "topk_303.jsonl"
PAIR_FILE = CKPT / "pairs_303w.jsonl"

IMG_U_DIR = Path(r"<PRF_ROOT>\archive\303\303u")

assert TOPK_FILE.exists(), f"Missing TOPK_FILE: {TOPK_FILE}"
assert IMG_U_DIR.exists(), f"Missing IMG_U_DIR: {IMG_U_DIR}"

def sha256_file(p: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()

# 1) build mapping sha256 -> path for 303u
u_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
    u_paths += list(IMG_U_DIR.glob(ext))
u_paths = sorted(set(u_paths))

print("[INFO] 303u images:", len(u_paths))

id2path = {}
for p in tqdm(u_paths, desc="hash 303u"):
    sid = sha256_file(p)
    id2path[sid] = str(p)

print("[INFO] id2path size:", len(id2path))
# quick sanity: show one
k0 = next(iter(id2path.keys()))
print("[SAMPLE]", k0, "->", id2path[k0])

# 2) expand topk -> pairs with candidate_path
pairs = 0
miss = 0

with TOPK_FILE.open("r", encoding="utf-8") as fin, PAIR_FILE.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="expand topk to pairs"):
        row = json.loads(line)
        qid = row["query_image_id"]
        qpath = row["query_path"]
        cand_ids = row["candidate_gallery_ids"]

        for cid in cand_ids:
            cpath = id2path.get(cid)
            if cpath is None:
                miss += 1
                continue
            fout.write(json.dumps({
                "schema": "pair_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": qpath,
                "candidate_path": cpath
            }, ensure_ascii=False) + "\n")
            pairs += 1

print("[OK] pair file written:", PAIR_FILE)
print("[OK] total pairs:", pairs)
print("[WARN] missing candidate_path:", miss)

# %% [cell 33]
import json, hashlib
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")

PAIR_IN  = CKPT / "pairs_303w.jsonl"          # 你已有（1515 pairs）
PAIR_OUT = CKPT / "pairs_303w_v2.jsonl"       # 新生成（带 candidate_path）

IMG_U_DIR = Path(r"<PRF_ROOT>\archive\303\303u")

assert PAIR_IN.exists(), f"Missing PAIR_IN: {PAIR_IN}"
assert IMG_U_DIR.exists(), f"Missing IMG_U_DIR: {IMG_U_DIR}"

def sha256_file(p: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()

# 1) build mapping sha256 -> path for 303u
u_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
    u_paths += list(IMG_U_DIR.glob(ext))
u_paths = sorted(set(u_paths))

print("[INFO] 303u images:", len(u_paths))

id2path = {}
for p in tqdm(u_paths, desc="hash 303u"):
    sid = sha256_file(p)
    id2path[sid] = str(p)

print("[INFO] id2path size:", len(id2path))

# 2) read old pairs and write new pairs with candidate_path
n_in = 0
n_out = 0
n_miss = 0

with PAIR_IN.open("r", encoding="utf-8") as fin, PAIR_OUT.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="upgrade pairs -> v2"):
        n_in += 1
        row = json.loads(line)

        # 兼容旧 schema：你现在的 keys 大概率是 query_image_id / candidate_image_id / query_path / candidate_path(没有)
        qid = row.get("query_image_id")
        cid = row.get("candidate_image_id")
        qpath = row.get("query_path")

        cpath = id2path.get(cid)
        if cpath is None:
            n_miss += 1
            continue

        fout.write(json.dumps({
            "schema": "pair_v2",
            "query_image_id": qid,
            "candidate_image_id": cid,
            "query_path": qpath,
            "candidate_path": cpath
        }, ensure_ascii=False) + "\n")
        n_out += 1

print("[OK] input pairs:", n_in)
print("[OK] output pairs_v2:", n_out)
print("[WARN] missing sha256->path:", n_miss)
print("[OUT]", PAIR_OUT)

# %% [cell 34]
# ===== Cell 15 (FINAL v4): B1 - use candidate_path from pairs (no filename matching) + data URL =====
import json
import base64
import requests
from pathlib import Path
from tqdm import tqdm

CKPT = Path(r"<PRF_CHECKPOINT_ROOT>\checkpoints")
IN_FILE  = CKPT / "pairs_303w_v2.jsonl"
OUT_FILE = CKPT / "b1_pair_review_303w_v2.jsonl"
ERR_FILE = CKPT / "b1_pair_review_303w_v2_errors.jsonl"

assert IN_FILE.exists(), f"Missing input file: {IN_FILE}"

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

SYSTEM_PROMPT = "你是图像配对审核专家，输出严格 JSON。"

USER_PROMPT = """
请对两张图像做配对审核，并只输出 JSON：
criteria: 五个维度，每个包含 value(0/1) 与 comment(中文一句建议)
- pixel_consistency
- same_base_image
- same_instance
- strong_structure
- weak_structure
final_recommendation: ACCEPT / REJECT / AMBIGUOUS
confidence: 0~1
summary_comment: 中文一句总结
"""

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
    assert "criteria" in res and isinstance(res["criteria"], dict)
    for k in ["pixel_consistency","same_base_image","same_instance","strong_structure","weak_structure"]:
        assert k in res["criteria"]
        assert res["criteria"][k]["value"] in [0,1]
        assert isinstance(res["criteria"][k]["comment"], str)
    assert res["final_recommendation"] in ["ACCEPT","REJECT","AMBIGUOUS"]
    conf = float(res["confidence"])
    assert 0.0 <= conf <= 1.0
    assert isinstance(res["summary_comment"], str)

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

    for line in tqdm(fin, desc="B1 8B review v2 (pairs_v2)"):
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
                "schema": "b1_pair_review_v2",
                "query_image_id": qid,
                "candidate_image_id": cid,
                "query_path": str(q_path),
                "candidate_path": str(c_path),
                "criteria": res["criteria"],
                "final_recommendation": res["final_recommendation"],
                "confidence": float(res["confidence"]),
                "summary_comment": res["summary_comment"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(key)
            n_ok += 1

        except Exception as e:
            ferr.write(json.dumps({
                "schema": "b1_pair_review_v2_error",
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
print("[OUT]", OUT_FILE)
print("[ERR]", ERR_FILE)

# %% [cell 35]

