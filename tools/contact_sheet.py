"""Planche contact de crops du portail, pour relire vite un lot de photos.

Usage : contact_sheet.py sortie.jpg chemin1.jpg[=legende] chemin2.jpg ...
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from roi import crop_gate  # noqa: E402

COLS, T = 5, 192


def main():
    out, items = sys.argv[1], sys.argv[2:]
    rows = (len(items) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * T, rows * (T + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for i, item in enumerate(items):
        path, _, caption = item.partition("=")
        x, y = (i % COLS) * T, (i // COLS) * (T + 14)
        sheet.paste(crop_gate(Image.open(path)).resize((T, T)), (x, y + 14))
        d.text((x + 2, y + 1), caption or Path(path).stem[-15:], fill="black")
    sheet.save(out, quality=90)


if __name__ == "__main__":
    main()
