"""One interface for every model: OpenAI and any OpenAI-compatible API (MiniMax, Ollama, OpenRouter)."""
import os
import time
from dataclasses import dataclass

import openai
from openai import OpenAI

RETRYABLE = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int


class LLMClient:
    """`structured` decides how the schema reaches the model:
    json_schema - provider-side structured output (strict JSON Schema);
    json_object - JSON mode, schema only in the prompt;
    none        - plain text, schema only in the prompt, JSON parsed from the answer.
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.model = cfg["model"]
        self.structured = cfg.get("structured", "json_schema")
        self.min_temperature = cfg.get("min_temperature", 0.0)
        self.max_retries = cfg.get("max_retries", 3)
        self.price_in = cfg["price_per_1m"]["input"]
        self.price_out = cfg["price_per_1m"]["output"]
        self.client = OpenAI(
            api_key=os.environ[cfg["api_key_env"]],
            base_url=cfg.get("base_url"),
            timeout=cfg.get("timeout_s", 60),
            max_retries=0,  # retries are ours, so they are counted and logged
        )

    def complete(self, messages: list[dict], json_schema: dict, temperature: float, top_p: float) -> LLMResponse:
        kwargs = dict(model=self.model, messages=messages,
                      temperature=max(temperature, self.min_temperature), top_p=top_p)
        if self.structured == "json_schema":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "resume", "strict": True, "schema": json_schema}}
        elif self.structured == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except RETRYABLE:
                if attempt > self.max_retries:
                    raise
                time.sleep(2 ** (attempt - 1))  # 1, 2, 4 s
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - start,
            attempts=attempt,
        )

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000
