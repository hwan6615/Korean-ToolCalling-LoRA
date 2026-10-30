import json
from collections import Counter
from datasets import load_dataset

ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

n_calls, n_tools, q_len = Counter(), Counter(), []
verbatim_ok = 0

for ex in ds:
    answers = json.loads(ex["answers"])
    tools = json.loads(ex["tools"])
    n_calls[len(answers)] += 1
    n_tools[len(tools)] += 1
    q_len.append(len(ex["query"]))

    query = ex["query"].lower()
    str_vals = [
        v for a in answers if isinstance(a.get("arguments"), dict)
        for v in a["arguments"].values() if isinstance(v, str)
    ]
    if all(v.lower() in query for v in str_vals):
        verbatim_ok += 1

print("정답 호출 개수 분포:", sorted(n_calls.items()))
print("제공 도구 개수 분포:", sorted(n_tools.items()))
print(f"질의 길이(문자) 평균 {sum(q_len)/len(q_len):.0f}, 최대 {max(q_len)}")
print(f"문자열 인자가 질의에 그대로 포함된 샘플: {verbatim_ok}/{len(ds)} ({verbatim_ok/len(ds):.1%})")