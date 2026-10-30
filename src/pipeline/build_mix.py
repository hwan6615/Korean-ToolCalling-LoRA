import json
import os
import random
import re
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.common import MAX_LEN, MODEL_ID, build_prompt, target_text

random.seed(11)
REPO = "palette-lab/songgot-tools-ko"
DATA_FILE = "songgot_tools_ko.jsonl"
N_KO = int(os.environ.get("N_KO", 2000))
HANGUL_FRAC = float(os.environ.get("HANGUL_FRAC", 0.7))
N_KO_HANGUL = round(N_KO * HANGUL_FRAC)
MAX_PER_TOOL = int(os.environ.get("MAX_PER_TOOL", 2))
OTHER_K = int(os.environ.get("OTHER_K", 40000))
FCB = "data/external/FunctionChat-Bench/data/FunctionChat-Singlecall.jsonl"
HANGUL = re.compile(r"[가-힣]")
OPTIONAL_HINT = re.compile(r"기본|선택|optional|default", re.I)
KEEP_EMPTY = os.environ.get("KEEP_EMPTY_WITH_PARAMS") == "1"
TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

xlam_lines = open("data/train_sft.jsonl", encoding="utf-8").read().splitlines()

# FunctionChat-Bench 평가 질의와 함수 이름 수집 (오염 방지용)
fcb_queries, fcb_names = [], set()
for line in open(FCB, encoding="utf-8"):
    r = json.loads(line)
    fcb_names.add(r["function_name"])
    fcb_queries += [q["content"] for q in r["query"]]
    for t in r["tools"]:
        for f in t["content"]:
            fcb_names.add(f["function"]["name"])


def grams(s, n=4):
    s = re.sub(r"\s+", "", s).lower()
    return {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}


fcb_grams = [grams(q) for q in fcb_queries]


def leaks(q):
    g = grams(q)
    return any(len(g & fg) / len(g | fg) >= 0.5 for fg in fcb_grams)


def prune(x):
    if isinstance(x, dict):
        return {k: prune(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [prune(v) for v in x]
    return x


def as_obj(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            return None
    return prune(x)


def parse_row(r):
    """(결과, None) 또는 (None, 탈락 사유)를 반환."""
    tool, call = as_obj(r.get("tool")), as_obj(r.get("call"))
    q = (r.get("query") or "").strip()
    if not isinstance(tool, dict) or not isinstance(call, dict) or not q:
        return None, "행 형식 이상"
    if len(q) > 300:
        return None, "질의 300자 초과"
    params = tool.get("parameters")
    if isinstance(params, str):
        params = as_obj(params)
    params = {} if params is None else params
    if not isinstance(params, dict) or not tool.get("name"):
        return None, "스키마 비표준"
    # 표준 형식(type/properties)이면 감싼 구조, 아니면 평평한 구조로 해석
    wrapped = params.get("type") == "object" and isinstance(params.get("properties"), dict)
    props = params["properties"] if wrapped else params
    if not all(isinstance(v, dict) for v in props.values()):
        return None, "스키마 비표준"
    for spec in props.values():
        t = spec.get("type")
        ok_type = t is None or isinstance(t, str) or (
            isinstance(t, list) and all(isinstance(x, str) for x in t)
        )
        if not ok_type:
            return None, "스키마 비표준(type)"
    if not isinstance(tool.get("name"), str):
        return None, "스키마 비표준"
    args = as_obj(call.get("arguments"))
    if call.get("name") != tool.get("name") or not isinstance(args, dict):
        return None, "호출과 도구 불일치"
    if not set(args) <= set(props):
        return None, "인자가 스키마에 없음"
    req_raw = params.get("required") if wrapped else None
    if isinstance(req_raw, list):
        req = [x for x in req_raw if isinstance(x, str)]
    else:  # required 정보가 없으면 정답 호출과 설명으로 추정
        req = [
            k
            for k in props
            if k in args and not OPTIONAL_HINT.search(str(props[k].get("description", "")))
        ]
    tool["parameters"] = {"properties": props, "required": req}
    svals = [v for v in args.values() if isinstance(v, str) and v]
    missing = [v for v in svals if v.lower() not in q.lower()]
    if missing:
        if any(HANGUL.search(v) for v in missing):
            return None, "한글 값이 질의에 그대로 없음"
        return None, "값이 질의에 없음(한글 아님)"
    has_hangul = any(HANGUL.search(v) for v in svals)
    return (tool, {"name": call["name"], "arguments": args}, q, has_hangul), None


def to_xlam(tool):
    schema = tool["parameters"]
    req = set(schema["required"])
    params = {}
    for k, spec in schema["properties"].items():
        t = spec.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), "string")
        if not isinstance(t, str):
            t = "string"
        desc = spec.get("description", "")
        p = {
            "description": desc if isinstance(desc, str) else "",
            "type": TYPE_MAP.get(t, t),
        }
        if k not in req:
            p["default"] = None
        params[k] = p
    return {"name": tool["name"], "description": str(tool.get("description", "")), "parameters": params}


path = hf_hub_download(REPO, DATA_FILE, repo_type="dataset")
why_count = Counter()
hangul_rows, other_rows, n_other, total = [], [], 0, 0
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        if total % 50000 == 0:
            print(f"  읽는 중 {total}행", end="\r")
        r = json.loads(line)
        p, why = parse_row(r)
        if p is None:
            why_count[why] += 1
            continue
        item = (r.get("domain"), *p)
        if p[3]:
            hangul_rows.append(item)
        else:
            n_other += 1
            if len(other_rows) < OTHER_K:
                other_rows.append(item)
            else:
                j = random.randrange(n_other)
                if j < OTHER_K:
                    other_rows[j] = item

print(f"\n{DATA_FILE}: {total}행")
print("탈락 사유:", dict(why_count.most_common()))
print(f"조건 통과: 한글 값 포함 {len(hangul_rows)}행, 한글 값 없음 {n_other}행 (이 중 {len(other_rows)}행 보관)")

pool_map, by_domain = {}, defaultdict(list)
for domain, tool, call, q, hg in hangul_rows + other_rows:
    name = tool["name"]
    if name in fcb_names or name in pool_map:
        continue
    pool_map[name] = to_xlam(tool)
    by_domain[domain].append(pool_map[name])
pool = list(pool_map.values())
print(f"도구 풀 {len(pool)}종, 도메인 {len(by_domain)}개")


def distractors(name, domain, k):
    out, used = [], {name}
    for _ in range(k * 8):
        if len(out) == k:
            break
        if by_domain.get(domain) and random.random() < 0.5:
            c = random.choice(by_domain[domain])
        else:
            c = random.choice(pool)
        if c["name"] in used:
            continue
        used.add(c["name"])
        out.append(c)
    return out


tok = AutoTokenizer.from_pretrained(MODEL_ID)
rows, flagged, too_long, capped, empty_dropped = [], 0, 0, 0, 0
seen_q, per_tool = set(), Counter()


def add(item):
    global flagged, too_long, capped, empty_dropped
    domain, tool, call, q, hg = item
    if q in seen_q:
        return False
    if not call["arguments"] and tool["parameters"]["properties"] and not KEEP_EMPTY:
        empty_dropped += 1
        return False
    if per_tool[tool["name"]] >= MAX_PER_TOOL:
        capped += 1
        return False
    if tool["name"] in fcb_names or leaks(q):
        flagged += 1
        return False
    tools = [to_xlam(tool)] + distractors(tool["name"], domain, random.randint(0, 5))
    random.shuffle(tools)
    prompt = build_prompt(tools, q)
    completion = [{"role": "assistant", "content": target_text([call])}]
    text = tok.apply_chat_template(prompt + completion, tokenize=False)
    if len(tok(text)["input_ids"]) > MAX_LEN:
        too_long += 1
        return False
    seen_q.add(q)
    per_tool[tool["name"]] += 1
    rows.append({"prompt": prompt, "completion": completion, "hg": hg})
    return True


random.shuffle(hangul_rows)
random.shuffle(other_rows)
n_h = 0
for item in hangul_rows:
    if n_h >= N_KO_HANGUL:
        break
    n_h += add(item)
for item in other_rows:
    if len(rows) >= N_KO:
        break
    add(item)

print(f"songgot {len(rows)}건 선택 (한글 값 행 {n_h}건 사용 / 가용 {len(hangul_rows)}건), 서로 다른 도구 {len(per_tool)}종")
print(f"FunctionChat과 겹쳐 제외 {flagged}건, 도구당 상한으로 건너뜀 {capped}건, 길이 초과 제외 {too_long}건, 빈 호출(파라미터 있는 도구) 제외 {empty_dropped}건")
xlam_rows = [json.loads(l) for l in xlam_lines]
songgot_rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
mix = songgot_rows + xlam_rows
random.shuffle(mix)
out_path = f"data/train_mix_{N_KO}.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for m in mix:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
hg_total = sum(bool(HANGUL.search(m["completion"][0]["content"])) for m in mix)
print(f"{out_path}: {len(mix)}건 (xLAM {len(xlam_rows)} + songgot {len(songgot_rows)}), 정답에 한글이 포함된 건 {hg_total}건")
print("\n샘플 (한글 값 3건, 그 외 2건):")
shown = {True: 0, False: 0}
for r in rows:
    lim = 3 if r["hg"] else 2
    if shown[r["hg"]] < lim:
        shown[r["hg"]] += 1
        print(" ", r["prompt"][1]["content"], "->", r["completion"][0]["content"])