"""Thinking-enabled requests to a locally deployed vLLM; preserve raw diagnostics."""
from __future__ import annotations
import time
import httpx
from llmgo.harness.agent import AskResult

class VllmAsker:
    def __init__(self, base_url, model, temperature=0.9, top_p=0.95, top_k=20,
                 presence_penalty=0.0, timeout_sec=540.0, transport=None):
        self.base_url = base_url.rstrip('/')
        self.model, self.temperature, self.top_p, self.top_k = model, temperature, top_p, top_k
        self.presence_penalty, self.timeout_sec, self.transport = presence_penalty, timeout_sec, transport

    def build_payload(self, prompt: str, max_tokens: int) -> dict:
        return {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': self.temperature, 'top_p': self.top_p, 'max_tokens': max_tokens,
                   'presence_penalty': self.presence_penalty, 'top_k': self.top_k,
                   'chat_template_kwargs': {'enable_thinking': True}}

    async def __call__(self, prompt: str, max_tokens: int) -> AskResult:
        payload = self.build_payload(prompt, max_tokens)
        started = time.monotonic()
        response = None
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=min(10, self.timeout_sec)), transport=self.transport) as client:
                response = await client.post(self.base_url + '/chat/completions', json=payload)
            raw = response.text
            if response.status_code >= 400:
                return AskResult('', 0, error=f'vLLM {response.status_code}: {raw}', raw_response=raw, tokens_exact=False,
                                 request_payload=payload, latency_ms=(time.monotonic() - started) * 1000, http_status=response.status_code)
            data = response.json()
            choices = data.get('choices') or []
            if not choices: raise ValueError('vLLM response has no choices')
            choice = choices[0]
            message = choice.get('message') or {}
            content = str(message.get('content') or '').strip()
            reasoning = str(message.get('reasoning_content') or message.get('reasoning') or '').strip()
            output = '\n\n'.join(part for part in (reasoning, content) if part)
            usage = data.get('usage') or {}
            exact = isinstance(usage.get('completion_tokens'), int)
            tokens = int(usage['completion_tokens']) if exact else max(1, len(output.encode('utf-8')))
            return AskResult(content, tokens, finish_reason=choice.get('finish_reason'), output=output,
                             reasoning=reasoning, usage=usage, tokens_exact=exact,
                             raw_response=raw, request_payload=payload, http_status=response.status_code,
                             latency_ms=(time.monotonic() - started) * 1000)
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            return AskResult('', 0, error=f'{type(exc).__name__}: {exc}', raw_response=response.text if response is not None else '', tokens_exact=False,
                             request_payload=payload, latency_ms=(time.monotonic() - started) * 1000,
                             http_status=response.status_code if response is not None else None)
