import json
import math
from collections import Counter

from src.common import parse_calls_lenient

RUNS = ["base", "base_hint", "lora_ep1", "lora_final", "base7b_hint", "qlora_final", "lora_rerun"]


def norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def args_of(c):
    a = c.get("arguments")
    return a if isinstance(a, dict) else {}


def key(c):
    return json.dumps(
        {"name": str(c.get("name")), "arguments": norm(args_of(c))},
        sort_keys=True,
        ensure_ascii=False,
    )


def names(calls):
    return Counter(str(c.get("name")) for c in calls)


def is_exact(pred, gold):
    return pred is not None and Counter(map(key, pred)) == Counter(map(key, gold))


def error_type(pred, gold):
    if pred is None:
        return "JSON 파싱 실패"
    if len(pred) != len(gold):
        return "호출 개수 불일치"
    if names(pred) != names(gold):
        return "도구 이름 불일치"
    pk = Counter((str(c["name"]), k) for c in pred for k in args_of(c))
    gk = Counter((str(c["name"]), k) for c in gold for k in args_of(c))
    extra, missing = pk - gk, gk - pk
    if extra and missing:
        return "인자 추가+누락"
    if extra:
        return "질의에 없는 인자 추가"
    if missing:
        return "인자 누락"
    return "인자 값 오류"


def mcnemar(a, b):
    ids = ok[a].keys() & ok[b].keys()
    x = sum(ok[a][i] and not ok[b][i] for i in ids)
    y = sum(ok[b][i] and not ok[a][i] for i in ids)
    n, k = x + y, min(x, y)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return x, y, p


rows = [json.loads(l) for l in open("data/clean.jsonl", encoding="utf-8")]
train_tools = {str(c["name"]) for r in rows if r["split"] == "train" for c in r["answers"]}
gold = {r["id"]: r for r in rows if r["split"] == "eval"}
seen = {i: all(str(c["name"]) in train_tools for c in r["answers"]) for i, r in gold.items()}
print(f"eval {len(gold)}건 중 정답 도구가 모두 train에 등장한 샘플: {sum(seen.values())}건")

ok, errs, examples = {}, {}, {}
for run in RUNS:
    try:
        outs = [json.loads(l) for l in open(f"outputs/eval/eval_out_{run}.jsonl", encoding="utf-8")]
    except FileNotFoundError:
        continue
    ok[run], errs[run] = {}, Counter()
    for o in outs:
        g = gold[o["id"]]["answers"]
        p = parse_calls_lenient(o["output"])
        e = is_exact(p, g)
        ok[run][o["id"]] = e
        if not e:
            et = error_type(p, g)
            errs[run][et] += 1
            if run == "lora_final":
                examples.setdefault(et, []).append((gold[o["id"]]["query_ko"], g, p))

print("\n완전 일치 (숫자 표기 정규화 후)")
for run, d in ok.items():
    def acc(sel):
        v = [x for i, x in d.items() if sel(i)]
        return f"{sum(v) / len(v):.1%} (n={len(v)})" if v else "-"
    print(f"  {run}: 전체 {acc(lambda i: True)} / 도구 seen {acc(lambda i: seen[i])} / unseen {acc(lambda i: not seen[i])}")

print("\n실패 유형:")
for run in errs:
    print(f"  {run}: {dict(errs[run].most_common())}")

for a, b in (("base_hint", "lora_final"), ("lora_ep1", "lora_final")):
    if a in ok and b in ok:
        x, y, p = mcnemar(a, b)
        print(f"\n{a} vs {b}: {a}만 정답 {x}건, {b}만 정답 {y}건, p={p:.4f}")

for et in ("인자 값 오류", "인자 누락", "질의에 없는 인자 추가"):
    print(f"\n[{et}] 예시 (lora_final)")
    for q, g, p in examples.get(et, [])[:3]:
        print(f"  질의: {q}")
        print(f"  정답: {json.dumps(g, ensure_ascii=False)[:250]}")
        print(f"  출력: {json.dumps(p, ensure_ascii=False)[:250]}")