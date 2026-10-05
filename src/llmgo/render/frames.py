"""用 Pillow 画一枚一枚的圆形棋子，并完整写下思考。"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from llmgo.env.notation import CHAR
from llmgo.env.xiangqi import Position

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
CELL = 62
MARGIN = 54
PIECE = 50


def _font(size: int) -> ImageFont.FreeTypeFont:
    for index in range(12):
        try:
            font = ImageFont.truetype(FONT_PATH, size, index=index)
        except OSError:
            break
        name = " ".join(font.getname())
        if "CJK SC" in name:
            return font
    return ImageFont.truetype(FONT_PATH, size, index=0)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if not text:
        return ["（这一手没有文字）"]
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            trial = current + char
            if font.getlength(trial) <= width and char != "\n":
                current = trial
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def draw_attempt(
    pos: Position,
    seat: str,
    uci: str | None,
    zh: str | None,
    reasoning: str,
    content: str,
    error: str | None,
) -> Image.Image:
    grid_w = CELL * 8
    grid_h = CELL * 9
    board_w = MARGIN * 2 + grid_w
    font = _font(22)
    small = _font(16)
    body = _font(18)
    header = [
        f"{'红方' if seat == 'red' else '黑方'}  {uci or '未走成'}  {zh or ''}",
        error or "这一手已被接受",
        "正式输出：",
    ]
    text_width = board_w - 48
    blocks = header + _wrap(content or "", body, text_width) + ["", "完整思考："] + _wrap(reasoning or "", body, text_width)
    line_h = 28
    text_h = 24 + line_h * len(blocks)
    image = Image.new("RGB", (board_w, MARGIN + grid_h + 36 + text_h), "#f4efe4")
    draw = ImageDraw.Draw(image)
    _draw_board(draw, pos, uci, font, small)
    y = MARGIN + grid_h + 28
    for line in blocks:
        draw.text((24, y), line, fill="#2a2118", font=body)
        y += line_h
    return image


def _draw_board(draw: ImageDraw.ImageDraw, pos: Position, uci: str | None, font, small) -> None:
    x0, y0 = MARGIN, MARGIN
    draw.rounded_rectangle((12, 12, MARGIN * 2 + CELL * 8 - 12, MARGIN + CELL * 9 + 18), 18, fill="#e7c07a")
    for rank in range(10):
        y = y0 + (9 - rank) * CELL
        draw.line((x0, y, x0 + 8 * CELL, y), fill="#6b3e1e", width=2)
    for file in range(9):
        x = x0 + file * CELL
        draw.line((x, y0, x, y0 + 4 * CELL), fill="#6b3e1e", width=2)
        draw.line((x, y0 + 5 * CELL, x, y0 + 9 * CELL), fill="#6b3e1e", width=2)
    draw.line((x0, y0 + 4 * CELL, x0, y0 + 5 * CELL), fill="#6b3e1e", width=2)
    draw.line((x0 + 8 * CELL, y0 + 4 * CELL, x0 + 8 * CELL, y0 + 5 * CELL), fill="#6b3e1e", width=2)
    _palace(draw, x0, y0, 0)
    _palace(draw, x0, y0, 7)
    river_y = y0 + 4.5 * CELL
    draw.text((x0 + CELL * 1.2, river_y - 16), "楚河", fill="#6b3e1e", font=font)
    draw.text((x0 + CELL * 5.2, river_y - 16), "汉界", fill="#6b3e1e", font=font)
    if uci and len(uci) == 4:
        for square, color in (((ord(uci[0]) - 97, int(uci[1])), "#2e7d32"), ((ord(uci[2]) - 97, int(uci[3])), "#1565c0")):
            cx = x0 + square[0] * CELL
            cy = y0 + (9 - square[1]) * CELL
            draw.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), outline=color, width=4)
    for file in range(9):
        draw.text((x0 + file * CELL - 6, 8), "abcdefghi"[file], fill="#6b3e1e", font=small)
    for item in pos.pieces():
        cx = x0 + item["file"] * CELL
        cy = y0 + (9 - item["rank"]) * CELL
        fill = "#f8e7c4" if item["color"] == "red" else "#f3e6cf"
        ring = "#a32020" if item["color"] == "red" else "#1d1a17"
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), fill=fill, outline=ring, width=3)
        draw.ellipse((cx - 19, cy - 19, cx + 19, cy + 19), outline=ring, width=1)
        char = CHAR[item["color"]][item["kind"]]
        draw.text((cx - 11, cy - 14), char, fill=ring, font=font)


def _palace(draw, x0: int, y0: int, top_rank: int) -> None:
    left = x0 + 3 * CELL
    right = x0 + 5 * CELL
    top = y0 + (9 - (top_rank + 2)) * CELL
    bottom = y0 + (9 - top_rank) * CELL
    draw.line((left, bottom, right, top), fill="#6b3e1e", width=2)
    draw.line((right, bottom, left, top), fill="#6b3e1e", width=2)
