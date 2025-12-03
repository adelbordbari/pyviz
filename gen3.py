#!/usr/bin/env python3
import argparse
import io
import keyword
import math
import random
import tokenize

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


def read_text(path):
    """Read the file as plain text, never executing it."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# --- Tokenization & grouping -------------------------------------------------


def tokenize_source(text):
    """
    Tokenize Python source and group tokens into semantic categories.
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

            weight = max(1, len(tok_str))
            result.append(
                {
                    "text": tok_str,
                    "group": group,
                    "weight": weight,
                }
            )
    except tokenize.TokenError:
        result.append({"text": text, "group": "other", "weight": len(text) or 1})

    return result


# --- Utility -----------------------------------------------------------------


PALETTE = {
    "background": "#111111",
    "keyword": "#312AF4",
    "name": "#C41F16",
    "string": "#46B2DD",
    "number": "#151F53",
    "comment": "#E5D7CC",
    "op": "#78E4C4",
    "other": "#EA8040",
}


def color_for_group(group):
    return PALETTE.get(group, PALETTE["other"])


def make_rng_from_text(text):
    seed = hash(text) & 0xFFFFFFFF
    return random.Random(seed)


# --- Curve layouts -----------------------------------------------------------


def build_orbit_layers(tokens, max_items, rng):
    """
    First panel: layered translucent circles ("orbits") around the centre.
    """
    if not tokens:
        return []

    total_weight = sum(t["weight"] for t in tokens) or 1.0

    elements = []
    count = 0
    for t in tokens:
        if count >= max_items:
            break
        group = t["group"]
        w = t["weight"]
        size_factor = 0.03 + 0.12 * (w / total_weight * len(tokens))
        size = max(0.01, min(0.22, size_factor))

        # polar coordinates around centre
        angle = rng.random() * 2 * math.pi
        radius = 0.1 + 0.35 * rng.random()
        cx = 0.5 + radius * math.cos(angle)
        cy = 0.5 + radius * math.sin(angle)

        alpha = 0.15 + 0.6 * rng.random()

        elements.append(
            {
                "type": "circle",
                "x": cx,
                "y": cy,
                "r": size,
                "group": group,
                "alpha": alpha,
            }
        )
        count += 1

    return elements


def build_ribbon_paths(tokens, max_items, rng):
    """
    Second panel: flowing cubic Bezier ribbons across the canvas.
    """
    if not tokens:
        return []

    elements = []
    count = 0
    for t in tokens:
        if count >= max_items:
            break

        group = t["group"]
        w = t["weight"]

        # start edge chosen by group
        side_choice = {
            "keyword": 0,
            "name": 1,
            "string": 2,
            "comment": 3,
        }.get(group, rng.randint(0, 3))

        if side_choice == 0:  # left → right
            x0, y0 = 0.0, rng.random()
            x3, y3 = 1.0, rng.random()
        elif side_choice == 1:  # bottom → top
            x0, y0 = rng.random(), 0.0
            x3, y3 = rng.random(), 1.0
        elif side_choice == 2:  # right → left
            x0, y0 = 1.0, rng.random()
            x3, y3 = 0.0, rng.random()
        else:  # top → bottom
            x0, y0 = rng.random(), 1.0
            x3, y3 = rng.random(), 0.0

        # control points with some curvature
        midx = (x0 + x3) / 2.0
        midy = (y0 + y3) / 2.0
        curve_strength = 0.3 + 0.3 * rng.random()
        angle = rng.random() * 2 * math.pi
        dx = math.cos(angle) * curve_strength
        dy = math.sin(angle) * curve_strength

        x1 = midx + dx
        y1 = midy + dy
        x2 = midx - dx
        y2 = midy - dy

        # clamp to [0,1]
        def clamp(v):
            return max(0.0, min(1.0, v))

        pts = [
            (x0, y0),
            (clamp(x1), clamp(y1)),
            (clamp(x2), clamp(y2)),
            (x3, y3),
        ]

        thickness = 0.002 + 0.012 * min(1.0, math.log(w + 1, 10) + rng.random())
        alpha = 0.2 + 0.5 * rng.random()

        elements.append(
            {
                "type": "bezier",
                "points": pts,
                "group": group,
                "alpha": alpha,
                "lw": thickness * 100,  # matplotlib linewidth (in points)
            }
        )
        count += 1

    return elements


def build_spiral_paths(tokens, max_items, rng):
    """
    Third panel: local spiral / vortex structures anchored by jittered points.
    """
    if not tokens:
        return []

    elements = []
    count = 0
    for t in tokens:
        if count >= max_items:
            break

        group = t["group"]
        w = t["weight"]

        # base centre jittered around canvas with slight bias
        base_x = rng.random()
        base_y = rng.random()

        # more important tokens get bigger spirals
        base_radius = 0.03 + 0.14 * (1.0 / (1.0 + math.exp(-0.3 * (w - 5))))

        turns = 1.0 + rng.random() * 2.0
        steps = 40

        pts = []
        for i in range(steps):
            t_rel = i / (steps - 1)
            angle = 2 * math.pi * turns * t_rel
            radius = base_radius * (0.3 + 0.7 * t_rel)
            x = base_x + radius * math.cos(angle)
            y = base_y + radius * math.sin(angle)
            pts.append((x, y))

        alpha = 0.15 + 0.55 * rng.random()
        lw = 0.3 + 1.4 * rng.random()

        elements.append(
            {
                "type": "polyline",
                "points": pts,
                "group": group,
                "alpha": alpha,
                "lw": lw,
            }
        )
        count += 1

    return elements


# --- Drawing -----------------------------------------------------------------


def draw_curve_panel(ax, elements, background="#050816"):
    ax.set_facecolor(background)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    for elem in elements:
        group = elem["group"]
        color = color_for_group(group)
        alpha = elem["alpha"]

        if elem["type"] == "circle":
            c = Circle(
                (elem["x"], elem["y"]),
                elem["r"],
                facecolor=color,
                edgecolor=None,
                alpha=alpha,
            )
            ax.add_patch(c)

        elif elem["type"] == "bezier":
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = elem["points"]
            verts = [
                (x0, y0),
                (x1, y1),
                (x2, y2),
                (x3, y3),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
            ]
            path = Path(verts, codes)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=elem["lw"],
                alpha=alpha,
            )
            ax.add_patch(patch)

        elif elem["type"] == "polyline":
            verts = elem["points"]
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            path = Path(verts, codes)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=elem["lw"],
                alpha=alpha,
            )
            ax.add_patch(patch)


def make_figure(tokens, title=None, max_items=300, text_for_seed=""):
    rng = make_rng_from_text(text_for_seed or "".join(t["text"] for t in tokens))

    orbits = build_orbit_layers(tokens, max_items=max_items // 3, rng=rng)
    ribbons = build_ribbon_paths(tokens, max_items=max_items // 2, rng=rng)
    spirals = build_spiral_paths(tokens, max_items=max_items // 2, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    draw_curve_panel(axes[0], orbits, background=PALETTE["background"])
    draw_curve_panel(axes[1], ribbons, background=PALETTE["background"])
    draw_curve_panel(axes[2], spirals, background=PALETTE["background"])

    if title:
        fig.suptitle(title)

    return fig


# --- CLI ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a Python source file with orbits, ribbons and spirals.\n"
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
        "--max-items",
        type=int,
        default=350,
        help="Maximum number of visual elements to generate (default: 350)",
    )

    args = parser.parse_args()

    text = read_text(args.source)
    tokens = tokenize_source(text)

    title = f"Curvy visualization of: {args.source}"
    fig = make_figure(tokens, title=title, max_items=args.max_items, text_for_seed=text)

    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
