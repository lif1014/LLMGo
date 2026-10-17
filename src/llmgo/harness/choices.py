"""Read explicit final answers without treating thoughts as choices."""
from __future__ import annotations
import json
import re


def split_thinking(content: str, reasoning: str = '') -> tuple[str, str, str | None]:
    """Handle server-separated reasoning, closing-tag output, and bare final JSON.

    Qwen's thinking mode sometimes streams reasoning and the final answer in the
    same ``content`` channel without emitting ``</think>``.  In that case the
    last ``{"move": "..."}`` object is the final answer; the preceding text is
    thinking and should not make the decision fall back to a random move.
    """
    if '</think>' in content:
        thoughts, final = content.rsplit('</think>', 1)
        thoughts = thoughts.removeprefix('<think>').strip()
        if thoughts and thoughts != reasoning:
            reasoning = '\n\n'.join(part for part in (reasoning, thoughts) if part)
        # A fresh unclosed block in the final portion is not a final answer.
        if '<think>' in final:
            return '', reasoning + '\n' + final, 'incomplete_thinking'
        return final.strip(), reasoning, None
    if '<think>' in content:
        return '', '\n\n'.join(part for part in (reasoning, content) if part), 'incomplete_thinking'
    matches = list(re.finditer(r'\{\s*"move"\s*:\s*"[^"]+"\s*\}', content, re.DOTALL))
    if matches:
        match = matches[-1]
        before = content[: match.start()].strip()
        after = content[match.end():].strip()
        reasoning = '\n\n'.join(part for part in (reasoning, before, after) if part)
        return match.group(0).strip(), reasoning, None
    return content.strip(), reasoning, None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate answer key')
        result[key] = value
    return result


def parse_move(text: str, allowed: set[str]) -> str | None:
    final, _, error = split_thinking(text)
    if error or not final:
        return None
    if final.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', final, re.DOTALL)
        if not match:
            return None
        final = match.group(1).strip()
    try:
        answer = json.loads(final, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        answer = final  # A bare UCI final answer is unambiguous too.
    if isinstance(answer, dict) and set(answer) == {'move'}:
        answer = answer['move']
    return answer if isinstance(answer, str) and answer in allowed else None


def parse_choice(text: str, allowed: set[str]) -> str | None:
    """Historical letter logs only; live decisions use parse_move."""
    final, _, error = split_thinking(text)
    if error:
        return None
    match = re.fullmatch(r'(?:(?:reply|answer)\s+)?([A-Za-z]|[0-9]+)', final, re.IGNORECASE)
    return next((value for value in allowed if match and value.upper() == match[1].upper()), None)
