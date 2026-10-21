"""Thinking-enabled requests to a locally deployed vLLM; preserve raw diagnostics."""
from __future__ import annotations
import asyncio
import json
import time
import httpx
from llmgo.harness.agent import AskResult
from llmgo.harness.choices import split_thinking
from llmgo.harness.streaming import SSEDecoder, StreamCancelled, StreamingText
from llmgo.harness.boundary import TokenBoundary


def _token_ids(value):
    if value is None:
        return []
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError('Invalid generated token IDs')
    return value


class VllmAsker:
    def __init__(self, base_url, model, temperature=0.9, top_p=0.95, top_k=20,
                 presence_penalty=0.0, timeout_sec=540.0, transport=None, tokenizer_path=None, seed=None):
        self.base_url = base_url.rstrip('/')
        self.model, self.temperature, self.top_p, self.top_k = model, temperature, top_p, top_k
        self.presence_penalty, self.timeout_sec, self.transport = presence_penalty, timeout_sec, transport
        self.boundary = TokenBoundary(tokenizer_path)
        self.seed = seed

    def build_payload(self, prompt: str, max_tokens: int) -> dict:
        return {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                   **({'seed': self.seed} if self.seed is not None else {}),
                   'temperature': self.temperature, 'top_p': self.top_p, 'max_tokens': max_tokens,
                   'presence_penalty': self.presence_penalty, 'top_k': self.top_k,
                   'skip_special_tokens': False, 'return_token_ids': True, 'include_reasoning': True,
                   'chat_template_kwargs': {'enable_thinking': True}}

    async def __call__(self, prompt: str, max_tokens: int) -> AskResult:
        payload = self.build_payload(prompt, max_tokens)
        started = time.monotonic()
        response = None
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=min(10, self.timeout_sec)), transport=self.transport, trust_env=False) as client:
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
            token_ids = _token_ids(choice.get('token_ids'))
            restored, boundary_error = self.boundary.restore(content, reasoning, token_ids)
            boundary_source = 'token_ids' if restored is not None else 'native_reasoning' if reasoning else 'content_tag' if '</think>' in content else 'standalone_or_missing'
            if restored is not None:
                content = restored
            output = '\n\n'.join(part for part in (reasoning, content) if part)
            content, reasoning, normalization_error = split_thinking(content, reasoning)
            usage = data.get('usage') or {}
            has_usage = isinstance(usage.get('completion_tokens'), int)
            exact = has_usage or bool(token_ids)
            tokens = int(usage['completion_tokens']) if has_usage else len(token_ids) if token_ids else max(1, len(output.encode('utf-8')))
            return AskResult(content, tokens, finish_reason=choice.get('finish_reason'), output=output,
                             reasoning=reasoning, usage=usage, tokens_exact=exact,
                             normalization_error=normalization_error,
                             token_ids=token_ids,
                             boundary_source=boundary_source, boundary_error=boundary_error,
                             tokenizer_info=self.boundary.windows.source,
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
        raw_parts, usage, token_ids = [], {}, []
        status, finish, error = None, None, None
        chunks, first_token_ms = 0, None
        is_sse = False
        received_done, received_choice = False, False
        recovered_boundary = False

        def result(failure=None):
            restored, boundary_error = self.boundary.restore(text.content, text.reasoning, token_ids)
            raw_content = restored if restored is not None else text.content
            source = ('token_ids' if recovered_boundary or restored is not None else 'native_reasoning' if text.reasoning
                      else 'content_tag' if '</think>' in raw_content else 'standalone_or_missing')
            content, reasoning, normalization_error = split_thinking(raw_content, text.reasoning)
            if not text.reasoning and '</think>' not in raw_content and text.shown['reasoning']:
                content, reasoning = '', raw_content.removeprefix('<think>')
                normalization_error = 'incomplete_thinking'
            output = '\n\n'.join(part for part in (text.reasoning, raw_content) if part)
            has_usage = isinstance(usage.get('completion_tokens'), int)
            exact = has_usage or bool(token_ids)
            tokens = int(usage['completion_tokens']) if has_usage else len(token_ids) if token_ids else len(output.encode('utf-8'))
            return AskResult(content, tokens, finish_reason=finish, error=failure,
                             output=output, reasoning=reasoning, usage=usage, tokens_exact=exact,
                             raw_response=b''.join(raw_parts).decode('utf-8', errors='replace'),
                             request_payload=payload, latency_ms=(time.monotonic() - started) * 1000,
                             http_status=status, normalization_error=normalization_error,
                             streamed=is_sse, stream_chunks=chunks, first_token_ms=first_token_ms,
                             token_ids=list(token_ids), boundary_source=source, boundary_error=boundary_error,
                             tokenizer_info=self.boundary.windows.source)

        async def dispatch(event):
            nonlocal usage, finish, chunks, first_token_ms, received_done, received_choice, recovered_boundary
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
                ids = _token_ids(choice.get('token_ids'))
                token_ids.extend(ids)
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
                delta = choice.get('delta') or {}
                if not isinstance(delta, dict): raise ValueError('Invalid stream delta')
                native = text.reasoning or delta.get('reasoning') or delta.get('reasoning_content')
                if self.boundary.hidden_end(delta, ids, native) and '</think>' not in text.content:
                    delta = {**delta, 'content': '</think>'}
                    recovered_boundary = True
                elif not native and self.boundary.end_id in ids and '</think>' not in text.content + (delta.get('content') or ''):
                    restored, _ = self.boundary.restore(text.content + (delta.get('content') or ''), '', token_ids)
                    if restored is not None:
                        text.content = restored
                        delta = {**delta, 'content': ''}
                        recovered_boundary = True
                for change in text.add(delta):
                    if first_token_ms is None and change['text']:
                        first_token_ms = (time.monotonic() - started) * 1000
                    chunks += 1
                    await emit({**change, 'chunk_index': chunks,
                                'elapsed_ms': (time.monotonic() - started) * 1000})

        try:
            # This deadline includes the entire response, not only gaps between chunks.
            async with asyncio.timeout(timeout_sec if timeout_sec is not None else self.timeout_sec):
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=min(10, self.timeout_sec)), transport=self.transport, trust_env=False) as client:
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
                            token_ids.extend(_token_ids(choice.get('token_ids')))
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
