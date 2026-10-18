"""Recover reasoning boundaries from model tokens, never from guessed move text."""
from llmgo.store.compact import TokenWindows


def visible_text(text):
    for marker in ('<think>', '</think>', '<|im_end|>', '<|endoftext|>'):
        text = text.replace(marker, '')
    return text.strip()


class TokenBoundary:
    def __init__(self, tokenizer_path=None):
        self.windows = TokenWindows(tokenizer_path)
        self.tokenizer = self.windows.tokenizer
        self.end_id = self.tokenizer.token_to_id('</think>') if self.tokenizer else None

    def hidden_end(self, delta, ids, native_reasoning):
        # vLLM can send the actual closing token as delta={}, token_ids=[id].
        # This is direct token evidence, including when skip_special_tokens=False.
        return (not native_reasoning and self.end_id is not None and ids == [self.end_id]
                and not delta.get('content'))

    def restore(self, content, reasoning, ids):
        if reasoning or '</think>' in content or not ids:
            return None, None
        if self.end_id is None:
            return None, self.windows.error
        if self.end_id not in ids:
            return None, None
        try:
            decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:
            return None, f'token_decode_error: {type(exc).__name__}: {exc}'
        if '</think>' not in decoded or visible_text(decoded) != visible_text(content):
            return None, 'token_text_mismatch: tokenizer does not reproduce received text'
        return decoded, None
