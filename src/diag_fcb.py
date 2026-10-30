import json
import os
import random
import re
from collections import defaultdict

from src.common import parse_calls_lenient

RUNS = ["base_hint", "lora_final", "base7b_hint", "qlora_final",
        "base_hint_note", "lora_final_note", "base7b_hint_note", "qlora_final_note", "lora_rerun"]
METRICS = ("name", "args_gold", "args_acc")
HANGUL = re.compile(r"[가-힣]")


def norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def load_json(s):
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return {}
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, (dict, list)) else {}


def gt_call(s):
    d = json.loads(s)
    return d["name"], load_json(d.get("arguments", {}))


def allowed(key, gold_val, acc):
    vals = [gold_val]
    if key in acc:
        a = acc[key]
        vals += a if isinstance(a, list) else [a]
    return [norm(v) for v in vals]


def judge(rec):
    pred = parse_calls_lenient(rec["output"])
    if pred is None:
        return {m: 0 for m in METRICS}, {}, {}, False
    name, gold = gt_call(rec["ground_truth"])
    acc = load_json(rec["acceptable"])
    acc = acc if isinstance(acc, dict) else {}
    c = pred[0] if len(pred) == 1 else {}
    pa = c.get("arguments", {})
    pa = pa if isinstance(pa, dict) else {}
    ok_name = len(pred) == 1 and str(c.get("name")) == name
    same = set(pa) == set(gold)
    ok_gold = ok_name and same and norm(pa) == norm(gold)
    ok_acc = ok_name and same and all(norm(pa[k]) in allowed(k, gold[k], acc) for k in gold)
    return {"name": int(ok_name), "args_gold": int(ok_gold), "args_acc": int(ok_acc)}, pa, gold, ok_name


rng = random.Random(0)


def boot(vals, n=2000):
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n)]


data = {}
for run in RUNS:
    path = f"outputs/eval/fcb_out_{run}.jsonl"
    if not os.path.exists(path):
        continue
    per_q = defaultdict(list)
    total = switched = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        s, pa, gold, ok_name = judge(r)
        per_q[(r["function_name"], r["serial_num"])].append(s)
        if ok_name:
            for k, gv in gold.items():
                if isinstance(gv, str) and HANGUL.search(gv) and isinstance(pa.get(k), str):
                    total += 1
                    switched += not HANGUL.search(pa[k])
    data[run] = {
        q: {m: sum(x[m] for x in v) / len(v) for m in METRICS} for q, v in per_q.items()
    }
    print(f"\n[{run}] 고유 질의 {len(per_q)}개")
    for m in METRICS:
        mean, lo, hi = boot([data[run][q][m] for q in data[run]])
        print(f"  {m}: {mean:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  한글 정답 값 {total}건 중 출력에 한글이 없음 {switched}건 ({switched / max(total, 1):.1%})")

print("\n모델 간 차이 (질의 단위 부트스트랩, 95% 구간)")
for a, b in (
    ("base_hint", "lora_final"),
    ("base7b_hint", "qlora_final"),
    ("lora_final", "qlora_final"),
    ("base7b_hint", "base7b_hint_note"),
    ("qlora_final", "qlora_final_note"),
    ("base7b_hint_note", "qlora_final_note"),
    ("lora_final", "lora_rerun"),
):
    if a in data and b in data:
        qs = sorted(data[a].keys() & data[b].keys())
        for m in METRICS:
            mean, lo, hi = boot([data[b][q][m] - data[a][q][m] for q in qs])
            print(f"  {b} - {a} / {m}: {mean:+.1%} [{lo:+.1%}, {hi:+.1%}]")

n = h = 0
for line in open("data/train_sft.jsonl", encoding="utf-8"):
    d = json.loads(line)
    n += 1
    h += bool(HANGUL.search(d["completion"][0]["content"]))
print(f"\n학습 정답 {n}건 중 한글이 포함된 정답: {h}건")