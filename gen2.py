#!/usr/bin/env python3
import argparse
import io
import keyword
import random
import tokenize

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_text(path):
    """Read the file as plain text, never executing it."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# --- Tokenization & grouping -------------------------------------------------


def tokenize_source(text):
    """
    Tokenize Python source and group tokens into semantic categories.

    This gives more structure than per-character: keywords, names, comments,
    strings, numbers, ops, etc.
    """
    result = []
    reader = io.StringIO(text).readline

    try:
        for tok in tokenize.generate_tokens(reader):
            tok_type, tok_str, start, end, line = tok
            if tok_type in (tokenize.ENCODING, tokenize.NL, tokenize.ENDMARKER):
                continue

            if tok_type == tokenize.COMMENT:
                group = "comment"
            elif tok_type == tokenize.STRING:
                group = "string"
            elif tok_type == tokenize.NUMBER:
                group = "number"
            elif tok_type == tokenize.OP:
                group = "op"
            elif tok_type == tokenize.NAME:
                if keyword.iskeyword(tok_str):
                    group = "keyword"
                else:
                    group = "name"
            else:
                group = "other"

            # weight: roughly "visual importance"
            weight = max(1, len(tok_str))
            result.append(
                {
                    "text": tok_str,
                    "group": group,
                    "weight": weight,
                }
            )
    except tokenize.TokenError:
        # On tokenize failure, fall back to treating all text as one big "other"
        result.append({"text": text, "group": "other", "weight": len(text) or 1})

    return result


# --- Rectangle layout --------------------------------------------------------


def build_rectangles(tokens, max_rects=400, min_size=0.02, margin=0.03, rng=None):
    """
    Recursively slice up a big rectangle based on tokens.

    - Which rectangle to split is chosen *by area*, so empty corners get refined.
    - Orientation & split ratio come from a RNG seeded by the file contents.
    - Each token influences several splits based on its weight.
    """
    if rng is None:
        rng = random.Random()

    rects = [
        {
            "x": margin,
            "y": margin,
            "w": 1.0 - 2 * margin,
            "h": 1.0 - 2 * margin,
            "group": "background",
            "token": None,
            "depth": 0,
        }
    ]
    if not tokens:
        return rects

    total_weight = sum(t["weight"] for t in tokens) or 1.0

    # Expand tokens into a sequence biased by weight, but capped to avoid explosion
    expanded = []
    target_len = max_rects * 2
    for t in tokens:
        share = t["weight"] / total_weight
        copies = max(1, int(share * target_len))
        for _ in range(copies):
            expanded.append(t)
    # shuffle for extra variety
    rng.shuffle(expanded)

    for t in expanded:
        if len(rects) >= max_rects:
            break

        # Choose a rectangle to split, probability ∝ area
        areas = [r["w"] * r["h"] for r in rects]
        total_area = sum(areas)
        if total_area <= 0:
            break

        pick = rng.random() * total_area
        acc = 0.0
        idx = 0
        for i, a in enumerate(areas):
            acc += a
            if acc >= pick:
                idx = i
                break

        rect = rects.pop(idx)

        if rect["w"] < min_size or rect["h"] < min_size:
            # Too small to split – keep it and move on
            rects.insert(idx, rect)
            continue

        # Orientation: a bit biased by token group, then jittered
        group = t["group"]
        if group in ("comment", "string"):
            # more horizontal
            orientation = 1 if rng.random() < 0.7 else 0
        elif group in ("keyword", "op"):
            # more vertical
            orientation = 0 if rng.random() < 0.7 else 1
        else:
            orientation = rng.randint(0, 1)

        # Split ratio: base depends on group, plus random jitter
        base = {
            "keyword": 0.35,
            "comment": 0.65,
            "string": 0.55,
            "number": 0.45,
            "name": 0.5,
            "op": 0.4,
        }.get(group, 0.5)
        jitter = (rng.random() - 0.5) * 0.3  # ±0.15
        ratio = min(0.8, max(0.2, base + jitter))

        def mk_rect(x, y, w, h):
            return {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "group": group,
                "token": t["text"],
                "depth": rect["depth"] + 1,
            }

        if orientation == 0:
            # vertical split
            w1 = rect["w"] * ratio
            w2 = rect["w"] - w1
            left = mk_rect(rect["x"], rect["y"], w1, rect["h"])
            right = mk_rect(rect["x"] + w1, rect["y"], w2, rect["h"])
            rects.insert(idx, right)
            rects.insert(idx, left)
        else:
            # horizontal split
            h1 = rect["h"] * ratio
            h2 = rect["h"] - h1
            bottom = mk_rect(rect["x"], rect["y"], rect["w"], h1)
            top = mk_rect(rect["x"], rect["y"] + h1, rect["w"], h2)
            rects.insert(idx, top)
            rects.insert(idx, bottom)

    return rects


# --- Palettes & drawing ------------------------------------------------------


PALETTES = [
    {
        "name": "midnight",
        "background": "#050816",
        "keyword": "#ff6b81",
        "name": "#4dabf7",
        "string": "#ffe066",
        "number": "#b197fc",
        "comment": "#868e96",
        "op": "#ff922b",
        "other": "#e9ecef",
    },
    {
        "name": "pastel",
        "background": "#f8f9fa",
        "keyword": "#ff6b6b",
        "name": "#4dabf7",
        "string": "#ffd43b",
        "number": "#9775fa",
        "comment": "#ced4da",
        "op": "#20c997",
        "other": "#343a40",
    },
    {
        "name": "forest",
        "background": "#102a43",
        "keyword": "#f6b93b",
        "name": "#1dd1a1",
        "string": "#ff9ff3",
        "number": "#54a0ff",
        "comment": "#576574",
        "op": "#ff6b6b",
        "other": "#c8d6e5",
    },
    {
        "name": "sunset",
        "background": "#1b1b2f",
        "keyword": "#ff7675",
        "name": "#74b9ff",
        "string": "#ffeaa7",
        "number": "#a29bfe",
        "comment": "#636e72",
        "op": "#fd79a8",
        "other": "#dfe6e9",
    },
    {
        "name": "mono",
        "background": "#ffffff",
        "keyword": "#111111",
        "name": "#333333",
        "string": "#666666",
        "number": "#999999",
        "comment": "#d0d0d0",
        "op": "#000000",
        "other": "#555555",
    },
]


def choose_palettes(text):
    """
    Pick 3 distinct palettes in a deterministic way based on the file contents.
    """
    seed = hash(text) & 0xFFFFFFFF
    rng = random.Random(seed)
    indices = list(range(len(PALETTES)))
    rng.shuffle(indices)
    chosen = [PALETTES[i] for i in indices[:3]]
    return chosen, rng


def draw_panel(ax, rects, palette, line_mode="thin"):
    """
    Draw rectangles using a given palette and line style.

    line_mode:
      - "thin": thin strokes, subtle
      - "thick": thick black strokes
      - "none": no edges
    """
    ax.set_facecolor(palette["background"])

    if not rects:
        return

    max_depth = max(r["depth"] for r in rects) or 1

    for r in rects:
        group = r["group"]
        color = palette.get(group, palette["other"])

        # Slight depth-based alpha variation
        depth_factor = (r["depth"] + 1) / (max_depth + 1)
        alpha = 0.4 + 0.6 * depth_factor

        if line_mode == "none":
            lw = 0.0
            edgecolor = None
        elif line_mode == "thick":
            lw = 1.5 + 1.5 * depth_factor
            edgecolor = "#000000"
        else:  # "thin"
            lw = 0.4 + 0.6 * depth_factor
            edgecolor = palette["background"]

        rect_patch = Rectangle(
            (r["x"], r["y"]),
            r["w"],
            r["h"],
            linewidth=lw,
            edgecolor=edgecolor,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect_patch)


def make_figure(rects, palettes, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Three different looks
    draw_panel(axes[0], rects, palettes[0], line_mode="thin")
    draw_panel(axes[1], rects, palettes[1], line_mode="thick")
    draw_panel(axes[2], rects, palettes[2], line_mode="none")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    if title:
        fig.suptitle(title)

    return fig


# --- CLI ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a Python source file as abstract rectangles.\n"
            "The file is never executed, only read as plain text."
        )
    )
    parser.add_argument("source", help="Path to the .py file to visualize")
    parser.add_argument(
        "-o",
        "--output",
        help="Output image filename (e.g. out.png). "
        "If omitted, the window is just shown.",
    )
    parser.add_argument(
        "--max-rects",
        type=int,
        default=400,
        help="Maximum number of rectangles to generate (default: 400)",
    )

    args = parser.parse_args()

    text = read_text(args.source)
    tokens = tokenize_source(text)

    palettes, rng = choose_palettes(text)
    rects = build_rectangles(
        tokens,
        max_rects=args.max_rects,
        rng=rng,
    )

    title = f"Visualization of: {args.source}"
    fig = make_figure(rects, palettes, title=title)

    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
