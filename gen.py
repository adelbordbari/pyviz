#!/usr/bin/env python3
import argparse
import string

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_text(path):
    """Read the file as plain text, never executing it."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def build_rectangles(text, max_rects=350, min_size=0.03, margin=0.03):
    """
    Recursively slice up a big rectangle based on the characters in `text`.

    Each character chooses which existing rectangle to split, the orientation
    of the split, and the split ratio. This gives a deterministic, maze-like,
    rectilinear layout driven purely by the text.
    """
    # initial rectangle inside a small margin
    rects = [{
        "x": margin,
        "y": margin,
        "w": 1.0 - 2 * margin,
        "h": 1.0 - 2 * margin,
        "char": " ",
        "depth": 0,
    }]
    if not text:
        return rects

    for i, ch in enumerate(text):
        if len(rects) >= max_rects:
            break

        code = ord(ch)
        idx = code % len(rects)
        rect = rects.pop(idx)

        if rect["w"] < min_size or rect["h"] < min_size:
            # put it back and skip splitting
            rects.insert(idx, rect)
            continue

        orientation = code % 2  # 0 vertical, 1 horizontal
        ratio_raw = (code % 70) + 15  # 15..84
        ratio = ratio_raw / 100.0

        if orientation == 0:
            # vertical split
            w1 = rect["w"] * ratio
            w2 = rect["w"] - w1
            left_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": w1,
                "h": rect["h"],
                "char": ch,
                "depth": rect["depth"] + 1,
            }
            right_rect = {
                "x": rect["x"] + w1,
                "y": rect["y"],
                "w": w2,
                "h": rect["h"],
                "char": ch,
                "depth": rect["depth"] + 1,
            }
            rects.insert(idx, right_rect)
            rects.insert(idx, left_rect)
        else:
            # horizontal split
            h1 = rect["h"] * ratio
            h2 = rect["h"] - h1
            bottom_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": rect["w"],
                "h": h1,
                "char": ch,
                "depth": rect["depth"] + 1,
            }
            top_rect = {
                "x": rect["x"],
                "y": rect["y"] + h1,
                "w": rect["w"],
                "h": h2,
                "char": ch,
                "depth": rect["depth"] + 1,
            }
            rects.insert(idx, top_rect)
            rects.insert(idx, bottom_rect)

    return rects


def char_group(ch):
    """Bucket characters into rough categories → used for color choices."""
    if ch in string.whitespace:
        return "whitespace"
    if ch in string.digits:
        return "digit"
    if ch in string.ascii_lowercase:
        return "lower"
    if ch in string.ascii_uppercase:
        return "upper"
    if ch in "()[]{}":
        return "bracket"
    if ch in ".,:;":
        return "punct"
    if ch in "+-*/%=&|^~<>":
        return "operator"
    return "other"


# palettes are hex colors
PALETTE_MOND = {
    "background": "#111111",
    "whitespace": "#f5f5f5",
    "upper": "#000000",
    "lower": "#ff3b30",
    "digit": "#D94C10",
    "bracket": "#46B2DD",
    "punct": "#151F53",
    "operator": "#C11E18",
    "other": "#312AF4",
}

PALETTE_STRIPES_BG = "#f7d600"

PALETTE_GREENISH = {
    "background": "#cddc39",
    "whitespace": "#f0f4c3",
    "upper": "#7cb342",
    "lower": "#8bc34a",
    "digit": "#558b2f",
    "bracket": "#2e7d32",
    "punct": "#aed581",
    "operator": "#689f38",
    "other": "#33691e",
}


def draw_style_stripes(ax, rects):
    """Left panel: yellow + hatching, lots of outlines."""
    ax.set_facecolor(PALETTE_STRIPES_BG)
    for r in rects:
        ch = r["char"]
        code = ord(ch)
        hatch_options = ["", "/", "\\\\", "//", "xx"]
        hatch = hatch_options[code % len(hatch_options)]
        rect_patch = Rectangle(
            (r["x"], r["y"]),
            r["w"],
            r["h"],
            linewidth=1.4,
            edgecolor="black",
            facecolor=PALETTE_STRIPES_BG,
            hatch=hatch,
        )
        ax.add_patch(rect_patch)


def draw_style_mondrian(ax, rects):
    """Middle panel: bold Mondrian-style primaries."""
    ax.set_facecolor(PALETTE_MOND["background"])
    for r in rects:
        group = char_group(r["char"])
        color = PALETTE_MOND.get(group, "#000000")
        rect_patch = Rectangle(
            (r["x"], r["y"]),
            r["w"],
            r["h"],
            linewidth=2.0,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(rect_patch)


def draw_style_green(ax, rects):
    """Right panel: greenish map-like look."""
    ax.set_facecolor(PALETTE_GREENISH["background"])
    for r in rects:
        ch = r["char"]
        group = char_group(ch)
        base_color = PALETTE_GREENISH.get(group, "#33691e")
        rect_patch = Rectangle(
            (r["x"], r["y"]),
            r["w"],
            r["h"],
            linewidth=1.2,
            edgecolor="black",
            facecolor=base_color,
        )
        ax.add_patch(rect_patch)


def make_figure(rects, title=None):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    draw_style_stripes(axes[0], rects)
    draw_style_mondrian(axes[1], rects)
    draw_style_green(axes[2], rects)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    if title:
        fig.suptitle(title)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a Python source file as abstract rectangles. "
                    "The file is never executed, only read as plain text."
    )
    parser.add_argument("source", help="Path to the .py file to visualize")
    parser.add_argument(
        "-o", "--output", help="Output image filename (e.g. out.png). "
                               "If omitted, the window is just shown."
    )
    parser.add_argument(
        "--max-rects",
        type=int,
        default=350,
        help="Maximum number of rectangles to generate (default: 350)",
    )

    args = parser.parse_args()

    text = read_text(args.source)
    rects = build_rectangles(text, max_rects=args.max_rects)

    title = f"Visualization of: {args.source}"
    fig = make_figure(rects, title=title)

    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
