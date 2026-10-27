"""Prompt building, extraction with schema validation and one repair attempt."""
import copy
import json
import re
import time
from pathlib import Path

from pydantic import ValidationError

from .llm import LLMClient
from .schema import Resume

# keywords OpenAI strict mode rejects; pydantic still enforces them after the call
_DROP = {"default", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "title"}


def strict_schema() -> dict:
    """Resume JSON Schema in the shape OpenAI strict structured output accepts."""
    def walk(node):
        if isinstance(node, dict):
            for key in _DROP & node.keys():
                del node[key]
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key, value in node.items():
                if key in ("properties", "$defs"):  # keys here are field names, not keywords
                    for sub in value.values():
                        walk(sub)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        return node
    return walk(copy.deepcopy(Resume.model_json_schema()))


def load_prompt(prompts_dir: Path, version: str) -> tuple[str, str]:
    """prompts/<version>.md holds '## system' and '## user' sections."""
    raw = (prompts_dir / f"{version}.md").read_text(encoding="utf-8")
    system = raw.split("## system", 1)[1].split("## user", 1)[0].strip()
    user = raw.split("## user", 1)[1].strip()
    return system, user


def build_messages(prompt: tuple[str, str], text: str, resume_date: str, schema: dict, guide: str) -> list[dict]:
    system, user = prompt
    fill = {"{schema}": json.dumps(schema, ensure_ascii=False), "{guide}": guide,
            "{resume_date}": resume_date, "{text}": text}
    for key, value in fill.items():  # plain replace: the schema itself is full of braces
        system, user = system.replace(key, value), user.replace(key, value)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json(raw: str) -> dict:
    """Strip reasoning blocks and code fences, then take the outermost JSON object."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def check(raw: str) -> tuple[dict | None, str | None]:
    try:
        return Resume.model_validate(parse_json(raw)).model_dump(mode="json"), None
    except (ValueError, ValidationError) as err:  # JSONDecodeError is a ValueError
        return None, str(err)[:1500]


def extract(client: LLMClient, messages: list[dict], schema: dict, sampling: dict) -> dict:
    """First call; if the answer is invalid, one repair call with the errors; otherwise 'failed'."""
    log = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "retries": 0}
    started = time.perf_counter()
    result, error, raw = None, None, ""
    for attempt in range(2):
        try:
            resp = client.complete(messages, schema, sampling["temperature"], sampling["top_p"])
        except Exception as err:  # network error after all retries, or a 4xx from the provider
            error = f"{type(err).__name__}: {err}"[:1500]
            break
        log["calls"] += 1
        log["retries"] += resp.attempts - 1
        log["input_tokens"] += resp.input_tokens
        log["output_tokens"] += resp.output_tokens
        raw = resp.text
        result, error = check(raw)
        if attempt == 0:
            log["valid_first"] = result is not None
            log["first_error"] = error
        if result is not None:
            break
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Відповідь не пройшла валідацію за схемою:\n" + error +
                                        "\nПоверни виправлений JSON повністю, без пояснень."}]
    log["latency_s"] = round(time.perf_counter() - started, 3)
    log.setdefault("valid_first", False)
    log.update(valid=result is not None, failed=result is None, repaired=result is not None and not log["valid_first"],
               error=error, output=result, raw_output=None if result is not None else raw[:4000],
               cost_usd=round(client.cost(log["input_tokens"], log["output_tokens"]), 6))
    return log
