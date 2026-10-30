import json
import re
from collections import Counter

from huggingface_hub import hf_hub_download

REPO = "palette-lab/songgot-tools-ko"
HANGUL = re.compile(r"[가-힣]")
path = hf_hub_download(REPO, "songgot_tools_ko.jsonl", repo_type="dataset")


def obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return x


cat, examples, rej = Counter(), [], 0
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    tool, call = obj(r.get("tool")), obj(r.get("call"))
    if not isinstance(tool, dict) or not isinstance(call, dict):
        cat["형식 이상"] += 1
        continue
    params, args = obj(tool.get("parameters")), obj(call.get("arguments"))
    if not isinstance(params, dict) or not isinstance(args, dict):
        cat["parameters/arguments 형식 이상"] += 1
        continue
    props = params.get("properties")
    if isinstance(props, dict) and set(args) <= set(props):
        cat["스키마 일치"] += 1
        continue
    if isinstance(props, dict):
        kind = "properties 있음, 인자 키 불일치"
    elif props is None:
        kind = "properties 키 없음"
    else:
        kind = f"properties 타입 {type(props).__name__}"
    cat[kind] += 1
    if any(isinstance(v, str) and HANGUL.search(v) for v in args.values()):
        cat[kind + " / 한글 값 포함"] += 1
    rej += 1
    if rej % 3000 == 1 and len(examples) < 8:
        examples.append(
            (
                kind,
                r.get("query"),
                json.dumps(params, ensure_ascii=False)[:300],
                json.dumps(args, ensure_ascii=False)[:200],
            )
        )

print("분류:")
for k, v in cat.most_common():
    print(f"  {k}: {v}")
print("\n탈락 예시:")
for kind, q, p, a in examples:
    print(f"- [{kind}] 질의: {q}\n  스키마: {p}\n  인자: {a}")

p2 = hf_hub_download(REPO, "tools.json", repo_type="dataset")
data = json.load(open(p2, encoding="utf-8"))
rows = data if isinstance(data, list) else data.get("data", [])
print(f"\ntools.json {len(rows)}행, 컬럼 {list(rows[0].keys())}")
for r in rows[:2]:
    print(json.dumps(r, ensure_ascii=False)[:600])