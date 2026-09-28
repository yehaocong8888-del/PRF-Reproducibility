{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "28c8930c-a20e-46be-8763-b8b12029cdf9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CFG] MOTHER_U: <FULL_PRF_ROOT>\\abcpipeline_full\\mother_U_v1.jsonl\n",
      "[CFG] PRED_V1 : <FULL_PRF_ROOT>\\YL\\HXYZ\\direct30b_pred_v1.jsonl\n",
      "[CFG] XRYZ    : <FULL_PRF_ROOT>\\YL\\XRYZ\n"
     ]
    }
   ],
   "source": [
    "import json, time, csv\n",
    "from pathlib import Path\n",
    "from collections import Counter, defaultdict\n",
    "from tqdm import tqdm\n",
    "\n",
    "ROOT = Path(r\"<FULL_PRF_ROOT>\")\n",
    "PIPE_DIR = ROOT / \"abcpipeline_full\"\n",
    "MOTHER_U = PIPE_DIR / \"mother_U_v1.jsonl\"\n",
    "\n",
    "HXYZ = ROOT / \"YL\" / \"HXYZ\"\n",
    "XRYZ = ROOT / \"YL\" / \"XRYZ\"\n",
    "XRYZ.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "PRED_V1 = HXYZ / \"direct30b_pred_v1.jsonl\"\n",
    "assert PRED_V1.exists(), PRED_V1\n",
    "assert MOTHER_U.exists(), MOTHER_U\n",
    "\n",
    "OUT_TABLE_CSV = XRYZ / \"ablation_metrics_v1.csv\"\n",
    "OUT_TABLE_JSON = XRYZ / \"ablation_metrics_v1.json\"\n",
    "OUT_PRED_JSONL = XRYZ / \"ablation_predictions_v1.jsonl\"\n",
    "\n",
    "DELTA_STRONG = -0.2\n",
    "\n",
    "def now_iso():\n",
    "    return time.strftime(\"%Y-%m-%dT%H:%M:%S\", time.localtime())\n",
    "\n",
    "def norm(x):\n",
    "    t = str(x).strip().upper()\n",
    "    if t in (\"ACCEPT\",\"REJECT\",\"AMBIGUOUS\"):\n",
    "        return t\n",
    "    if \"ACCEPT\" in t: return \"ACCEPT\"\n",
    "    if \"REJECT\" in t: return \"REJECT\"\n",
    "    return \"AMBIGUOUS\"\n",
    "\n",
    "def key_of(qid, cid):\n",
    "    return f\"{int(qid)}_{int(cid)}\"\n",
    "\n",
    "print(\"[CFG] MOTHER_<LOCAL_DRIVE_U>\", MOTHER_U)\n",
    "print(\"[CFG] PRED_V1 :\", PRED_V1)\n",
    "print(\"[CFG] XRYZ    :\", XRYZ)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e6501e5a-8b0a-4c60-be06-ea711e16e073",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] mother_U indexed: 80431\n"
     ]
    }
   ],
   "source": [
    "# Build an index: key -> full mother_U row (contains A/B/C decisions)\n",
    "idx = {}\n",
    "with MOTHER_U.open(\"r\", encoding=\"utf-8\") as <LOCAL_DRIVE_F>\n",
    "    for line in <LOCAL_DRIVE_F>\n",
    "        if not line.strip():\n",
    "            continue\n",
    "        j = json.loads(line)\n",
    "        k = key_of(j[\"query_image_id\"], j[\"candidate_image_id\"])\n",
    "        idx[k] = j\n",
    "\n",
    "print(\"[INFO] mother_U indexed:\", len(idx))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "45d8dec2-71fd-4d2e-80ae-6cc33e2871af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] sample (from core experiment): 4294\n"
     ]
    }
   ],
   "source": [
    "sample = []\n",
    "with PRED_V1.open(\"r\", encoding=\"utf-8\") as <LOCAL_DRIVE_F>\n",
    "    for line in <LOCAL_DRIVE_F>\n",
    "        if line.strip():\n",
    "            sample.append(json.loads(line))\n",
    "\n",
    "print(\"[INFO] sample (from core experiment):\", len(sample))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "96a08294-c1e0-4d83-98d4-8b1ce770aec8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_votes(row):\n",
    "    \"\"\"\n",
    "    Return (voteA, voteB, voteC) each in {ACCEPT, REJECT, AMBIGUOUS}\n",
    "    Using fields observed in your mother_U snapshot:\n",
    "      A.vote: ACCEPT/AMBIGUOUS (rarely REJECT)\n",
    "      B.final: ACCEPT/REJECT/AMBIGUOUS\n",
    "      C.decision: AMBIGUOUS (but keep)\n",
    "    \"\"\"\n",
    "    A = row.get(\"A\", {})\n",
    "    B = row.get(\"B\", {})\n",
    "    C = row.get(\"C\", {})\n",
    "\n",
    "    vA = norm(A.get(\"vote\", A.get(\"decision\", \"AMBIGUOUS\")))\n",
    "    vB = norm(B.get(\"final\", \"AMBIGUOUS\"))\n",
    "    vC = norm(C.get(\"decision\", \"AMBIGUOUS\"))\n",
    "    return vA, vB, vC\n",
    "\n",
    "def majority_2of3(votes):\n",
    "    # 2-of-3 majority, ties => AMBIGUOUS\n",
    "    c = Counter(votes)\n",
    "    for lab in (\"ACCEPT\",\"REJECT\",\"AMBIGUOUS\"):\n",
    "        if c[lab] >= 2:\n",
    "            return lab\n",
    "    return \"AMBIGUOUS\"\n",
    "\n",
    "def ablate_decision(row, variant: str) -> str:\n",
    "    \"\"\"\n",
    "    Produce ablated decision WITHOUT teacher.\n",
    "    Variants:\n",
    "      - BASELINE: mother_U.final_decision (for reference)\n",
    "      - NO_GATE: ignore force_dispute; use 2-of-3 directly\n",
    "      - NO_C: use votes from A,B only (2-of-2; disagree -> AMBIGUOUS)\n",
    "      - ONLY_B: use B.final\n",
    "      - ONE_OF_THREE_ACCEPT: if any vote==ACCEPT -> ACCEPT; elif any vote==REJECT -> REJECT; else AMBIGUOUS\n",
    "    \"\"\"\n",
    "    vA, vB, vC = get_votes(row)\n",
    "\n",
    "    if variant == \"NO_GATE\":\n",
    "        return majority_2of3([vA, vB, vC])\n",
    "\n",
    "    if variant == \"NO_C\":\n",
    "        # 2-of-2 on A,B; if disagree -> AMBIGUOUS\n",
    "        if vA == vB:\n",
    "            return vA\n",
    "        return \"AMBIGUOUS\"\n",
    "\n",
    "    if variant == \"ONLY_B\":\n",
    "        return vB\n",
    "\n",
    "    if variant == \"ONE_OF_THREE_ACCEPT\":\n",
    "        if \"ACCEPT\" in (vA, vB, vC):\n",
    "            return \"ACCEPT\"\n",
    "        if \"REJECT\" in (vA, vB, vC):\n",
    "            return \"REJECT\"\n",
    "        return \"AMBIGUOUS\"\n",
    "\n",
    "    raise ValueError(\"unknown variant\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "84f987a6-8f35-4bd2-b3fa-6adfc25faa77",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Join sample with mother_U: 100%|████████████████████████████| 4294/4294 [00:00<00:00, 186695.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] joined rows: 4294\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[OUT] metrics json -> <FULL_PRF_ROOT>\\YL\\XRYZ\\ablation_metrics_v1.json\n",
      "[OUT] metrics csv  -> <FULL_PRF_ROOT>\\YL\\XRYZ\\ablation_metrics_v1.csv\n",
      "[OUT] preds jsonl  -> <FULL_PRF_ROOT>\\YL\\XRYZ\\ablation_predictions_v1.jsonl\n",
      "\n",
      "[ABLATION SUMMARY]\n",
      "PRF_BASELINE agree_rate= 0.6772240335351654 dist_pred= {'REJECT': 2268, 'ACCEPT': 1442, 'AMBIGUOUS': 584}\n",
      "NO_GATE agree_rate= 0.45808104331625527 dist_pred= {'REJECT': 1358, 'ACCEPT': 927, 'AMBIGUOUS': 2009}\n",
      "NO_C agree_rate= 0.42594317652538427 dist_pred= {'REJECT': 1198, 'ACCEPT': 927, 'AMBIGUOUS': 2169}\n",
      "ONLY_B agree_rate= 0.6744294364229158 dist_pred= {'REJECT': 2231, 'ACCEPT': 1452, 'AMBIGUOUS': 611}\n",
      "ONE_OF_THREE_ACCEPT agree_rate= 0.7135537959944108 dist_pred= {'REJECT': 2371, 'ACCEPT': 1618, 'AMBIGUOUS': 305}\n"
     ]
    }
   ],
   "source": [
    "variants = [\"NO_GATE\", \"NO_C\", \"ONLY_B\", \"ONE_OF_THREE_ACCEPT\"]\n",
    "\n",
    "def compute_metrics(rows):\n",
    "    # rows: list of dict with fields {direct, pred, prf}\n",
    "    n = len(rows)\n",
    "    dist_pred = Counter()\n",
    "    dist_direct = Counter()\n",
    "    agree = 0\n",
    "    for r in rows:\n",
    "        p = norm(r[\"pred\"])\n",
    "        d = norm(r[\"direct\"])\n",
    "        dist_pred[p] += 1\n",
    "        dist_direct[d] += 1\n",
    "        if p == <LOCAL_DRIVE_D>\n",
    "            agree += 1\n",
    "    return {\n",
    "        \"n\": n,\n",
    "        \"agree\": agree,\n",
    "        \"agree_rate\": agree/n if n else None,\n",
    "        \"dist_pred\": dict(dist_pred),\n",
    "        \"dist_direct\": dict(dist_direct),\n",
    "    }\n",
    "\n",
    "# prepare baseline direct decisions\n",
    "records = []\n",
    "for r in tqdm(sample, desc=\"Join sample with mother_U\", ncols=100):\n",
    "    k = key_of(r[\"query_image_id\"], r[\"candidate_image_id\"])\n",
    "    m = idx.get(k)\n",
    "    if m is None:\n",
    "        continue\n",
    "    records.append({\n",
    "        \"key\": k,\n",
    "        \"query_image_id\": r[\"query_image_id\"],\n",
    "        \"candidate_image_id\": r[\"candidate_image_id\"],\n",
    "        \"query_path\": r[\"query_path\"],\n",
    "        \"candidate_path\": r[\"candidate_path\"],\n",
    "        \"direct\": norm(r[\"direct30b\"][\"final_decision\"]),   # V1 direct\n",
    "        \"prf_final\": norm(m.get(\"final_decision\")),\n",
    "        \"prf_source\": m.get(\"final_source\"),\n",
    "        \"force_dispute\": bool(m.get(\"force_dispute\", False)),\n",
    "        \"c_delta\": m.get(\"C\", {}).get(\"delta\", None),\n",
    "        \"mother_row\": m,   # keep for ablation\n",
    "    })\n",
    "\n",
    "print(\"[INFO] joined rows:\", len(records))\n",
    "\n",
    "# run ablations\n",
    "all_out = []\n",
    "summary = {\n",
    "    \"schema\": \"ablation_metrics_v1\",\n",
    "    \"timestamp\": now_iso(),\n",
    "    \"inputs\": {\n",
    "        \"mother_U\": str(MOTHER_U),\n",
    "        \"pred_v1\": str(PRED_V1),\n",
    "        \"variants\": variants,\n",
    "    },\n",
    "    \"results\": {}\n",
    "}\n",
    "\n",
    "# baseline metrics (PRF vs Direct) on same joined set\n",
    "base_rows = [{\"pred\": r[\"prf_final\"], \"direct\": r[\"direct\"]} for r in records]\n",
    "summary[\"results\"][\"PRF_BASELINE\"] = compute_metrics(base_rows)\n",
    "\n",
    "for var in variants:\n",
    "    rows_var = []\n",
    "    for r in records:\n",
    "        pred = ablate_decision(r[\"mother_row\"], var)\n",
    "        rows_var.append({\"pred\": pred, \"direct\": r[\"direct\"]})\n",
    "        all_out.append({\n",
    "            \"schema\": \"ablation_pred_v1\",\n",
    "            \"variant\": var,\n",
    "            \"key\": r[\"key\"],\n",
    "            \"query_image_id\": r[\"query_image_id\"],\n",
    "            \"candidate_image_id\": r[\"candidate_image_id\"],\n",
    "            \"direct_v1\": r[\"direct\"],\n",
    "            \"prf_final\": r[\"prf_final\"],\n",
    "            \"ablated_pred\": pred,\n",
    "            \"prf_source\": r[\"prf_source\"],\n",
    "            \"force_dispute\": r[\"force_dispute\"],\n",
    "            \"c_delta\": r[\"c_delta\"],\n",
    "            \"query_path\": r[\"query_path\"],\n",
    "            \"candidate_path\": r[\"candidate_path\"],\n",
    "        })\n",
    "    summary[\"results\"][var] = compute_metrics(rows_var)\n",
    "\n",
    "# write json + csv\n",
    "with OUT_TABLE_JSON.open(\"w\", encoding=\"utf-8\") as <LOCAL_DRIVE_F>\n",
    "    json.dump(summary, f, ensure_ascii=False, indent=2)\n",
    "\n",
    "with OUT_TABLE_CSV.open(\"w\", encoding=\"utf-8\", newline=\"\") as <LOCAL_DRIVE_F>\n",
    "    w = csv.writer(f)\n",
    "    w.writerow([\"variant\",\"n\",\"agree\",\"agree_rate\",\"pred_ACCEPT\",\"pred_REJECT\",\"pred_AMBIGUOUS\"])\n",
    "    for name, stat in summary[\"results\"].items():\n",
    "        dist = stat[\"dist_pred\"]\n",
    "        w.writerow([\n",
    "            name, stat[\"n\"], stat[\"agree\"], stat[\"agree_rate\"],\n",
    "            dist.get(\"ACCEPT\",0), dist.get(\"REJECT\",0), dist.get(\"AMBIGUOUS\",0)\n",
    "        ])\n",
    "\n",
    "with OUT_PRED_JSONL.open(\"w\", encoding=\"utf-8\") as <LOCAL_DRIVE_F>\n",
    "    for j in all_out:\n",
    "        f.write(json.dumps(j, ensure_ascii=False) + \"\\n\")\n",
    "\n",
    "print(\"[OUT] metrics json ->\", OUT_TABLE_JSON)\n",
    "print(\"[OUT] metrics csv  ->\", OUT_TABLE_CSV)\n",
    "print(\"[OUT] preds jsonl  ->\", OUT_PRED_JSONL)\n",
    "\n",
    "print(\"\\n[ABLATION SUMMARY]\")\n",
    "for name, stat in summary[\"results\"].items():\n",
    "    print(name, \"agree_rate=\", stat[\"agree_rate\"], \"dist_pred=\", stat[\"dist_pred\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bab3d5cf-877a-401f-b4e6-bdec7de69fbd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6dbc3975-86e5-45c1-8e1a-72feb2e31d6a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "903aef46-6de6-42e1-af71-1f2223f2c581",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78139d33-c752-4264-b701-5aac2177f89a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11aee36f-d9b8-4bef-b78b-996716d43296",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4604d925-d6d5-4c9e-b018-fdc2716190e0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "rtx5090_final (CUDA13.1)",
   "language": "python",
   "name": "rtx5090_final"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
