import json
import os
import ast
from collections import defaultdict

from src.common import parse_calls_lenient

RUNS = ["base_hint", "lora_final", "base7b_hint", "qlora_final",
        "base_hint_note", "lora_final_note", "base7b_hint_note", "qlora_final_note", "lora_rerun"]


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


BAD = []


def load_json(s):
    if s is None:
        return {}
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, (dict, list)) else {}
    except (ValueError, SyntaxError):
        BAD.append(s)
        return {}


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
        return {"valid": False, "name": False, "args_gold": False, "args_acc": False}, None
    name, gold = gt_call(rec["ground_truth"])
    acc = load_json(rec["acceptable"])
    acc = acc if isinstance(acc, dict) else {}
    one = len(pred) == 1
    c = pred[0] if one else {}
    pa = c.get("arguments", {})
    pa = pa if isinstance(pa, dict) else {}
    ok_name = one and str(c.get("name")) == name
    same_keys = set(pa) == set(gold)
    ok_gold = ok_name and same_keys and norm(pa) == norm(gold)
    ok_acc = ok_name and same_keys and all(norm(pa[k]) in allowed(k, gold[k], acc) for k in gold)
    return {"valid": True, "name": ok_name, "args_gold": ok_gold, "args_acc": ok_acc}, pa


def rate(rs, k):
    return f"{sum(r[k] for r in rs) / len(rs):.1%}"


for run in RUNS:
    path = f"outputs/eval/fcb_out_{run}.jsonl"
    if not os.path.exists(path):
        continue
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    scored = [(r, *judge(r)) for r in recs]
    groups = defaultdict(list)
    for r, s, _ in scored:
        groups["전체"].append(s)
        groups[f"도구목록 {r['tools_type']}"].append(s)
        groups[f"추출유형 {r['extraction_type']}"].append(s)
    print(f"\n[{run}] n={len(recs)}")
    for g in sorted(groups, key=lambda x: (x != "전체", x)):
        v = groups[g]
        print(
            f"  {g} (n={len(v)}): JSON 유효 {rate(v,'valid')} / 도구명 {rate(v,'name')}"
            f" / 인자(정답만) {rate(v,'args_gold')} / 인자(허용값 포함) {rate(v,'args_acc')}"
        )

    if run in ("lora_final", "qlora_final"):
        bad = [(r, pa) for r, s, pa in scored if s["name"] and not s["args_acc"]]
        print(f"\n  도구명은 맞고 인자가 틀린 사례 {len(bad)}건 중 12건:")
        for r, pa in bad[:12]:
            print(f"    [{r['extraction_type']}] 질의: {r['query']}")
            print(f"    정답: {gt_call(r['ground_truth'])[1]} / 허용: {r['acceptable']}")
            print(f"    출력: {pa}")
        print(f"\n해석하지 못한 acceptable 값: {len(BAD)}건")
        for s in BAD[:5]:
            print("  ", repr(s)[:200])