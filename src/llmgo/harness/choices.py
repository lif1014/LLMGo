"""字母選單。字母固定，查過的工具整段拿掉。"""

from __future__ import annotations

import re

TOOL_ORDER = ("legal", "hanging", "king", "material", "preview", "piece")
TOOL_LETTER = {
    "legal": "A",
    "hanging": "B",
    "king": "C",
    "material": "D",
    "piece": "E",
    "preview": "F",
}
LETTER_TOOL = {letter: name for name, letter in TOOL_LETTER.items()}

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def option_labels(count: int) -> list[str]:
    """一題裡的選項字母。超過 26 個時接著用 1、2。"""
    if count <= len(_ALPHABET):
        return list(_ALPHABET[:count])
    extra = [str(index) for index in range(1, count - len(_ALPHABET) + 1)]
    return list(_ALPHABET) + extra


def parse_choice(text: str, allowed: set[str]) -> str | None:
    """從模型回覆取出一個允許的字母。幾個都出現時視為沒選。"""
    if not text or not allowed:
        return None
    by_key = {item.upper(): item for item in allowed}
    found = []
    for item in allowed:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])", text, re.IGNORECASE):
            found.append(item)
    unique = {item.upper() for item in found}
    if len(unique) == 1:
        return by_key[next(iter(unique))]
    return None


def available_tools(used: set[str]) -> list[str]:
    """查過的工具消失。寫進便籤滿兩次後，只留 piece。"""
    if len(used) >= 2:
        return ["piece"]
    return [name for name in TOOL_ORDER if name == "piece" or name not in used]


def blackboard(notes: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for name in ("legal", "hanging", "king", "material", "preview"):
        lines.extend(notes.get(name, []))
    return lines
