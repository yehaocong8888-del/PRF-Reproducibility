import os, json, random
from pathlib import Path
from collections import defaultdict, Counter

TRAIN = Path(os.environ.get("TRAIN_JSONL", str(Path.home() / "PRF_WORK_v1/dataset/train.jsonl")))
OUT   = Path(os.environ.get("KEEP_OUT",   str(Path.home() / "PRF_WORK_v1/keep_indices_trainA_v2.json")))

assert TRAIN.exists(), f"Missing TRAIN_JSONL: {TRAIN}"
OUT.parent.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "42"))
K_PER_QUERY = int(os.environ.get("K_PER_QUERY", "100"))

P_U = float(os.environ.get("P_U", "1.0"))   # U=1.0
P_A = float(os.environ.get("P_A", "0.3"))   # A=0.3
P_R = float(os.environ.get("P_R", "0.3"))   # R=0.3

random.seed(SEED)

def prob(dec: str) -> float:
    if dec == "U": return P_U
    if dec == "A": return P_A
    if dec == "R": return P_R
    # 未知标签：保守给 0.3
    return 0.3

print("[READ-ONLY] Only reads train.jsonl and writes keep_indices json. No other files modified.")
print("[IN]  TRAIN:", TRAIN)
print("[OUT] KEEP :", OUT)
print("[CFG] K_PER_QUERY =", K_PER_QUERY, "| P(U,A,R) =", (P_U, P_A, P_R), "| SEED =", SEED)

N = 0
keep = []
per_q = defaultdict(int)

cnt_all = Counter()
cnt_keep = Counter()
stats = Counter()

with TRAIN.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        # train.jsonl 的真实字段：query_image_id / candidate_image_id / target.decision
        qid = r.get("query_image_id", None)
        tgt = (r.get("target") or {})
        dec = tgt.get("decision", None)
        if qid is None or dec is None:
            stats["skip_bad_row"] += 1
            N += 1
            continue

        dec = str(dec).strip().upper()
        cnt_all[dec] += 1

        # per-query cap
        if per_q[qid] >= K_PER_QUERY:
            stats["drop_query_cap"] += 1
            N += 1
            continue

        p = prob(dec)
        if random.random() <= p:
            keep.append(N)
            per_q[qid] += 1
            cnt_keep[dec] += 1
        else:
            stats["drop_prob"] += 1

        N += 1

keep_obj = {
    "schema": "keep_trainA_v2",
    "train_jsonl": str(TRAIN),
    "seed": SEED,
    "K_PER_QUERY": K_PER_QUERY,
    "P_U": P_U, "P_A": P_A, "P_R": P_R,
    "train_lines": N,
    "keep_count": len(keep),
    "keep_ratio": (len(keep)/max(1,N)),
    "decision_all": dict(cnt_all),
    "decision_keep": dict(cnt_keep),
    "stats": dict(stats),
    "keep_indices": keep,
}

OUT.write_text(json.dumps(keep_obj, ensure_ascii=False), encoding="utf-8")

print("\n[TRAIN] lines =", N)
print("[ALL ] decision dist:", dict(cnt_all))
print("[KEEP] keep_count =", len(keep), f"({len(keep)/max(1,N):.2%})")
print("[KEEP] decision dist:", dict(cnt_keep))
print("[STATS]", dict(stats))
print("\n[DONE] wrote:", OUT)
