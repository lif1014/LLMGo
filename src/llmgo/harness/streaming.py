"""Incremental SSE framing and live views of Qwen reasoning/final output."""
from __future__ import annotations
import asyncio
import codecs
import re
from llmgo.harness.choices import split_thinking


class StreamCancelled(asyncio.CancelledError):
    def __init__(self, partial):
        super().__init__('request_cancelled')
        self.partial = partial


class SSEDecoder:
    """Accept split UTF-8, CRLF, comments and multi-line data events."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer = ''
        self.data = []

    def feed(self, chunk: bytes, *, final=False) -> list[str]:
        self.buffer += self.decoder.decode(chunk, final=final)
        events = []
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.removesuffix('\r')
            if not line:
                if self.data:
                    events.append('\n'.join(self.data)); self.data = []
            elif line.startswith('data:'):
                self.data.append(line[5:].removeprefix(' '))
        # An incomplete frame is not dispatched on EOF.
        return events


class StreamingText:
    """Emit channel patches; replacements handle provisional mixed-content views."""
    def __init__(self):
        self.content = ''
        self.reasoning = ''
        self.shown = {'reasoning': '', 'content': ''}

    def add(self, delta: dict) -> list[dict]:
        self.content += delta.get('content') or ''
        self.reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
        if '</think>' in self.content:
            final, thought, _ = split_thinking(self.content, self.reasoning)
        elif self.reasoning:
            thought, final = self.reasoning, self.content
        elif self.content.lstrip().startswith(('{', '```')) or re.fullmatch(r'[a-h](?:[1-8](?:[a-h](?:[1-8][qrbn]?)?)?)?', split_thinking(self.content)[0]):
            thought, final = '', self.content
        else:
            thought, final = self.content.removeprefix('<think>'), ''
            for marker in ('<think>', '</think>'):
                for size in range(1, len(marker)):
                    if thought.endswith(marker[:size]):
                        thought = thought[:-size]
                        break
        changes = []
        for channel, value in (('reasoning', thought), ('content', final)):
            old = self.shown[channel]
            if value != old:
                replace = not value.startswith(old)
                changes.append({'channel': channel, 'text': value if replace else value[len(old):], 'replace': replace})
                self.shown[channel] = value
        return changes
