"""Starter harness. Replace freely during Phase 2 research."""
from pathlib import Path


def build_messages(fen, legal_san, skills, default_messages):
    messages = [dict(message) for message in default_messages]
    # Keep the original prompt and append a locally maintained skill card.
    card = (Path(__file__).parent/'skills/position-checklist.md').read_text()
    messages[-1]['content'] += '\n\nAdditional chess guidance:\n' + card
    return messages


# Optional alternative: def build_messages_with_history(request): ...
# request contains fen, legal_san, skills, default_messages, initial_fen,
# moves_uci and moves_san. Return ordinary chat messages.
