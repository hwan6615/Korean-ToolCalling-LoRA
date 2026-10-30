import json
import random
import re
from collections import defaultdict
from datasets import load_dataset

random.seed(42)
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")


def string_values(answers):
    vals = []
    for a in answers:
        args = a.get("arguments")
        if isinstance(args, dict):
            vals += [v for v in args.values() if isinstance(v, str) and v]
    return vals


def in_query(v, q):
    pat = r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])"
    return re.search(pat, q, re.I) is not None


def is_literal(v, q):
    quoted = re.search(r"['\"‘“]" + re.escape(v) + r"['\"’”]", q) is not None
    special = re.search(r"[A-Z0-9._@/:-]", v) is not None
    return quoted or special


pool = defaultdict(list)
seen_queries = set()
for ex in ds:
    q = ex["query"].strip()
    if q in seen_queries or len(q) > 300:
        continue
    answers = json.loads(ex["answers"])
    tools = json.loads(ex["tools"])
    if not (1 <= len(answers) <= 3) or len(tools) > 4:
        continue
    vals = string_values(answers)
    if not all(in_query(v, q) and is_literal(v, q) for v in vals):
        continue
    seen_queries.add(q)
    pool[len(answers)].append(
        {"id": ex["id"], "query": q, "answers": answers, "tools": tools}
    )

quota = {1: 0.5, 2: 0.4, 3: 0.1}
TOTAL, N_EVAL = 3300, 300
rows = []
for k, frac in quota.items():
    random.shuffle(pool[k])
    n = round(TOTAL * frac)
    n_eval = round(N_EVAL * frac)
    for i, r in enumerate(pool[k][:n]):
        r["split"] = "eval" if i < n_eval else "train"
        rows.append(r)

random.shuffle(rows)
with open("data/selected.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("필터 통과 풀 크기:", {k: len(v) for k, v in pool.items()})
print("저장:", len(rows), {s: sum(r["split"] == s for r in rows) for s in ("train", "eval")})