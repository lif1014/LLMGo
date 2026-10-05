"""把逐手图片收成 GIF。没有 ffmpeg 时这也是录像。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def write_gif(frames: list[Image.Image], path: Path) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = frames
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=900,
        loop=0,
        optimize=False,
    )
