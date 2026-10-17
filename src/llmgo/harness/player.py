"""Thinking-enabled requests to a locally deployed vLLM; preserve raw diagnostics."""
from __future__ import annotations
import asyncio
import json
import time
import httpx
from llmgo.harness.agent import AskResult
from llmgo.harness.choices import split_thinking
from llmgo.harness.streaming import SSEDecoder, StreamCancelled, StreamingText

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
            content, reasoning, normalization_error = split_thinking(content, reasoning)
            usage = data.get('usage') or {}
            exact = isinstance(usage.get('completion_tokens'), int)
            tokens = int(usage['completion_tokens']) if exact else max(1, len(output.encode('utf-8')))
            return AskResult(content, tokens, finish_reason=choice.get('finish_reason'), output=output,
                             reasoning=reasoning, usage=usage, tokens_exact=exact,
                             normalization_error=normalization_error,
                             raw_response=raw, request_payload=payload, http_status=response.status_code,
                             latency_ms=(time.monotonic() - started) * 1000)
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            return AskResult('', 0, error=f'{type(exc).__name__}: {exc}', raw_response=response.text if response is not None else '', tokens_exact=False,
                             request_payload=payload, latency_ms=(time.monotonic() - started) * 1000,
                             http_status=response.status_code if response is not None else None)

    def build_stream_payload(self, prompt: str, max_tokens: int) -> dict:
        return {**self.build_payload(prompt, max_tokens), 'stream': True,
                'stream_options': {'include_usage': True}}

    async def stream(self, prompt: str, max_tokens: int, emit, *, timeout_sec=None) -> AskResult:
        payload = self.build_stream_payload(prompt, max_tokens)
        started = time.monotonic()
        text, frames = StreamingText(), SSEDecoder()
        raw_parts, usage = [], {}
        status, finish, error = None, None, None
        chunks, first_token_ms = 0, None
        is_sse = False
        received_done, received_choice = False, False

        def result(failure=None):
            content, reasoning, normalization_error = split_thinking(text.content, text.reasoning)
            if not text.reasoning and '</think>' not in text.content and text.shown['reasoning']:
                content, reasoning = '', text.content.removeprefix('<think>')
                normalization_error = 'incomplete_thinking'
            output = '\n\n'.join(part for part in (text.reasoning, text.content) if part)
            exact = isinstance(usage.get('completion_tokens'), int)
            tokens = int(usage['completion_tokens']) if exact else len(output.encode('utf-8'))
            return AskResult(content, tokens, finish_reason=finish, error=failure,
                             output=output, reasoning=reasoning, usage=usage, tokens_exact=exact,
                             raw_response=b''.join(raw_parts).decode('utf-8', errors='replace'),
                             request_payload=payload, latency_ms=(time.monotonic() - started) * 1000,
                             http_status=status, normalization_error=normalization_error,
                             streamed=is_sse, stream_chunks=chunks, first_token_ms=first_token_ms)

        async def dispatch(event):
            nonlocal usage, finish, chunks, first_token_ms, received_done, received_choice
            if event.strip() == '[DONE]':
                received_done = True
                return
            data = json.loads(event)
            if not isinstance(data, dict):
                raise ValueError('vLLM stream event must be an object')
            if data.get('error'):
                raise ValueError('vLLM stream error: ' + str(data['error']))
            if data.get('usage'):
                if not isinstance(data['usage'], dict): raise ValueError('Invalid stream usage')
                usage = data['usage']
            for choice in data.get('choices') or []:
                if not isinstance(choice, dict): raise ValueError('Invalid stream choice')
                if choice.get('index', 0) != 0:
                    continue
                received_choice = True
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
                delta = choice.get('delta') or {}
                if not isinstance(delta, dict): raise ValueError('Invalid stream delta')
                for change in text.add(delta):
                    if first_token_ms is None and change['text']:
                        first_token_ms = (time.monotonic() - started) * 1000
                    chunks += 1
                    await emit({**change, 'chunk_index': chunks,
                                'elapsed_ms': (time.monotonic() - started) * 1000})

        try:
            # This deadline includes the entire response, not only gaps between chunks.
            async with asyncio.timeout(timeout_sec if timeout_sec is not None else self.timeout_sec):
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=min(10, self.timeout_sec)), transport=self.transport) as client:
                    async with client.stream('POST', self.base_url + '/chat/completions', json=payload) as response:
                        status = response.status_code
                        is_sse = 'text/event-stream' in response.headers.get('content-type', '').lower()
                        if response.status_code >= 400:
                            raw_parts.append(await response.aread())
                            return result(f'vLLM {status}: ' + raw_parts[-1].decode('utf-8', errors='replace'))
                        if not is_sse:
                            # Some compatible test/proxy servers ignore stream=true.
                            raw_parts.append(await response.aread())
                            data = json.loads(raw_parts[-1])
                            choice = (data.get('choices') or [])[0]
                            message = choice.get('message') or {}
                            text.content = str(message.get('content') or '')
                            text.reasoning = str(message.get('reasoning_content') or message.get('reasoning') or '')
                            usage = data.get('usage') or {}
                            finish = choice.get('finish_reason')
                            answer = result()
                            answer.streamed = False
                            return answer
                        async for part in response.aiter_bytes():
                            raw_parts.append(part)
                            for event in frames.feed(part):
                                await dispatch(event)
                                if received_done:
                                    break
                            if received_done:
                                break
                        if not received_done:
                            for event in frames.feed(b'', final=True):
                                await dispatch(event)
                        if not received_choice or finish is None:
                            error = 'incomplete_stream: no final finish_reason'
        except TimeoutError:
            error = 'request_timeout'
        except asyncio.CancelledError:
            raise StreamCancelled(result('request_cancelled'))
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        return result(error)
