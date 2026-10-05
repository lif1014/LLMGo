"""通过本机 vLLM 的 OpenAI 接口生成。每个座位一个实例，请求之间不共享消息。"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmgo.runtime.player import StreamEvent, seat_messages
from llmgo.runtime.prompt import Observation


class VllmPlayer:
    def __init__(
        self,
        seat: str,
        base_url: str,
        model: str,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
    ):
        self.seat = seat
        self.name = f"vllm-{seat}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty

    async def play(self, obs: Observation) -> AsyncIterator[StreamEvent]:
        if obs.seat != self.seat:
            raise RuntimeError(f"{self.name} 收到了 {obs.seat} 的局面")
        payload = {
            "model": self.model,
            "messages": seat_messages(obs),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "stream": True,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        finish = None
        url = f"{self.base_url}/chat/completions"
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"vLLM {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if reasoning is None:
                        reasoning = delta.get("reasoning")
                    content = delta.get("content")
                    if reasoning:
                        yield StreamEvent("reasoning", reasoning)
                    if content:
                        yield StreamEvent("content", content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        yield StreamEvent("done", finish_reason=finish or "stop")
