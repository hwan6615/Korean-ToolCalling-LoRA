import json
import re
from collections import Counter
import os

# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
USE_4BIT = os.environ.get("USE_4BIT") == "1"
RUN_NAME = os.environ.get("RUN_NAME", "qwen1.5b-lora")
OUT_DIR = f"outputs/{RUN_NAME}"

MAX_LEN = 2048

SYSTEM = (
    "You are a function-calling assistant. Given the available tools (JSON) and a user "
    "request in Korean, respond with ONLY a JSON array of tool calls, each of the form "
    '{"name": ..., "arguments": {...}}. Copy argument values exactly as they appear in '
    "the request. Do not add explanations."
)


HINT = (
    "\n\nOutput format example (two calls): "
    '[{"name": "tool_a", "arguments": {"x": 1}}, {"name": "tool_b", "arguments": {"y": "z"}}]'
)


NOTE = (
    "\n\nUnless there is a special reason, do not translate the user's Korean "
    "expressions into English when generating parameter values."
)


def build_prompt(tools, query, hint=False, note=False):
    system = (
        SYSTEM
        + (HINT if hint else "")
        + (NOTE if note else "")
        + "\n\nTools:\n"
        + json.dumps(tools, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def target_text(answers):
    return json.dumps(answers, ensure_ascii=False)


def parse_calls(text):
    m = re.search(r"\[.*\]", text.strip(), re.S)
    if not m:
        return None
    try:
        calls = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list) or not all(
        isinstance(c, dict) and "name" in c for c in calls
    ):
        return None
    return calls


def call_key(c):
    return json.dumps(
        {"name": str(c.get("name")), "arguments": c.get("arguments", {})},
        sort_keys=True,
        ensure_ascii=False,
    )


def score(pred, gold):
    if pred is None:
        return {"valid": False, "name": False, "exact": False}
    return {
        "valid": True,
        "name": Counter(str(c["name"]) for c in pred)
        == Counter(str(c["name"]) for c in gold),
        "exact": Counter(map(call_key, pred)) == Counter(map(call_key, gold)),
    }
    
def parse_calls_lenient(text):
    m = re.search(r"\[.*\]", text.strip(), re.S)
    if not m:
        return None
    raw, data = m.group(0), None
    for cand in (raw, "[" + raw + "]"):
        try:
            data = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, list):
        return None
    if len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], dict):
        data = [data]
    calls = []
    for item in data:
        if isinstance(item, dict) and "name" in item:
            calls.append(item)
        elif (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], dict)
        ):
            calls.append({"name": item[0], "arguments": item[1]})
        else:
            return None
    return calls

def load_model(device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {}
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device_map, **kwargs
    )