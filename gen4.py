#!/usr/bin/env python3
import argparse
import io
import keyword
import random
import tokenize

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyBboxPatch,
    PathPatch,
)
from matplotlib.path import Path


def read_text(path):
    """Read the file as plain text, never executing it."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# --- Tokenization & grouping -------------------------------------------------


def tokenize_source(text):
    """Tokenize Python source and group tokens into semantic categories."""
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


# --- Organic layout (rectangle grid used as soft cells) ----------------------


def build_cells(tokens, max_cells=400, min_size=0.02, margin=0.03, rng=None):
    """Slice up a big rectangle based on tokens (same idea as your rect grid).

    We still use a recursive subdivision because it produces nice structure,
    but we only treat the result as *cells* that will host curvy shapes.
    """
    if rng is None:
        rng = random.Random()

    cells = [
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
        return cells

    total_weight = sum(t["weight"] for t in tokens) or 1.0

    # Expand tokens into a sequence biased by weight, but capped to avoid explosion
    expanded = []
    target_len = max_cells * 2
    for t in tokens:
        share = t["weight"] / total_weight
        copies = max(1, int(share * target_len))
        for _ in range(copies):
            expanded.append(t)
    rng.shuffle(expanded)

    for t in expanded:
        if len(cells) >= max_cells:
            break

        # Choose a cell to split, probability ∝ area
        areas = [r["w"] * r["h"] for r in cells]
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

        cell = cells.pop(idx)

        if cell["w"] < min_size or cell["h"] < min_size:
            # Too small to split – keep it and move on
            cells.insert(idx, cell)
            continue

        # Orientation: similar bias rules as your original
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

        def mk_cell(x, y, w, h):
            return {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "group": group,
                "token": t["text"],
                "depth": cell["depth"] + 1,
            }

        if orientation == 0:
            # vertical split
            w1 = cell["w"] * ratio
            w2 = cell["w"] - w1
            left = mk_cell(cell["x"], cell["y"], w1, cell["h"])
            right = mk_cell(cell["x"] + w1, cell["y"], w2, cell["h"])
            cells.insert(idx, right)
            cells.insert(idx, left)
        else:
            # horizontal split
            h1 = cell["h"] * ratio
            h2 = cell["h"] - h1
            bottom = mk_cell(cell["x"], cell["y"], cell["w"], h1)
            top = mk_cell(cell["x"], cell["y"] + h1, cell["w"], h2)
            cells.insert(idx, top)
            cells.insert(idx, bottom)

    return cells


# --- Palettes & RNG ----------------------------------------------------------


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
    """Pick 3 distinct palettes in a deterministic way based on the file contents."""
    seed = hash(text) & 0xFFFFFFFF
    rng = random.Random(seed)
    indices = list(range(len(PALETTES)))
    rng.shuffle(indices)
    chosen = [PALETTES[i] for i in indices[:3]]
    return chosen, rng


# --- Drawing -----------------------------------------------------------------


def _pick_shape_kind(group, rng):
    """Return a small enum describing which kind of shape to draw."""
    base = {
        "keyword": ["ring", "arc", "blob"],
        "name": ["circle", "pill", "blob"],
        "string": ["circle", "circle", "ring", "swirl"],
        "number": ["pill", "blob"],
        "comment": ["soft_rect", "arc"],
        "op": ["arc", "swirl", "ring"],
        "other": ["blob", "circle"],
        "background": ["soft_rect"],
    }.get(group, ["blob"])

    return rng.choice(base)


def _draw_curve_in_cell(ax, cell, color, rng, alpha=0.9, lw=1.0):
    """Add a single smooth Bezier curve snaking through the cell."""
    x = cell["x"]
    y = cell["y"]
    w = cell["w"]
    h = cell["h"]

    # Start & end points on opposite edges
    if rng.random() < 0.5:
        start = (x, y + rng.random() * h)
        end = (x + w, y + rng.random() * h)
    else:
        start = (x + rng.random() * w, y)
        end = (x + rng.random() * w, y + h)

    # Two control points to bend the curve
    ctrl1 = (x + rng.random() * w, y + rng.random() * h)
    ctrl2 = (x + rng.random() * w, y + rng.random() * h)

    verts = [start, ctrl1, ctrl2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)


def draw_panel_curvy(ax, cells, palette, rng, mode="mixed"):
    """Draw curvy shapes using a given palette.

    mode:
      - "bubbles": mostly circles / rings
      - "flow": emphasise curves between cells
      - "mixed": a bit of everything
    """
    ax.set_facecolor(palette["background"])

    if not cells:
        return

    max_depth = max(c["depth"] for c in cells) or 1

    for c in cells:
        group = c["group"]
        color = palette.get(group, palette["other"])

        # Slight depth-based alpha variation
        depth_factor = (c["depth"] + 1) / (max_depth + 1)
        fill_alpha = 0.35 + 0.45 * depth_factor

        cx = c["x"] + c["w"] * 0.5
        cy = c["y"] + c["h"] * 0.5
        size = min(c["w"], c["h"])

        pad = size * 0.1
        inner_w = max(0.0, c["w"] - 2 * pad)
        inner_h = max(0.0, c["h"] - 2 * pad)

        # Decide which base shape we want
        if mode == "bubbles":
            shape_kind = rng.choice(["circle", "circle", "ring", "pill"])
        elif mode == "flow":
            shape_kind = rng.choice(["blob", "arc", "swirl"])
        else:
            shape_kind = _pick_shape_kind(group, rng)

        # Draw the main filled shape (if any)
        patch = None
        if shape_kind == "circle":
            radius = size * rng.uniform(0.3, 0.5)
            patch = Circle(
                (cx, cy),
                radius=radius,
                facecolor=color,
                edgecolor="none",
                alpha=fill_alpha,
            )
        elif shape_kind == "ring":
            radius = size * rng.uniform(0.3, 0.5)
            patch = Circle(
                (cx, cy),
                radius=radius,
                facecolor="none",
                edgecolor=color,
                linewidth=0.4 + 0.8 * depth_factor,
                alpha=fill_alpha + 0.2,
            )
        elif shape_kind == "ellipse":
            patch = Ellipse(
                (cx, cy),
                width=inner_w * rng.uniform(0.5, 1.0),
                height=inner_h * rng.uniform(0.3, 0.9),
                angle=rng.uniform(0, 180),
                facecolor=color,
                edgecolor="none",
                alpha=fill_alpha,
            )
        elif shape_kind == "pill":
            # A very rounded rectangle
            patch = FancyBboxPatch(
                (c["x"] + pad, c["y"] + pad),
                inner_w,
                inner_h * rng.uniform(0.4, 1.0),
                boxstyle="round,pad=0.02,rounding_size={}".format(size * 0.5),
                facecolor=color,
                edgecolor="none",
                alpha=fill_alpha,
            )
        elif shape_kind in ("blob", "soft_rect"):
            # Squishy rounded rect
            rx = c["x"] + pad * rng.uniform(0.5, 1.4)
            ry = c["y"] + pad * rng.uniform(0.5, 1.4)
            rw = max(0.0, inner_w * rng.uniform(0.7, 1.0))
            rh = max(0.0, inner_h * rng.uniform(0.7, 1.0))
            patch = FancyBboxPatch(
                (rx, ry),
                rw,
                rh,
                boxstyle="round,pad=0.02,rounding_size={}".format(
                    size * rng.uniform(0.15, 0.4)
                ),
                facecolor=color,
                edgecolor="none",
                alpha=fill_alpha,
            )

        if patch is not None:
            ax.add_patch(patch)

        # Occasionally overlay a curve or swirl
        if shape_kind in ("arc", "swirl") or (
            mode in ("flow", "mixed") and rng.random() < 0.35
        ):
            curve_alpha = 0.4 + 0.4 * depth_factor
            lw = 0.6 + 1.0 * depth_factor
            _draw_curve_in_cell(ax, c, color=color, rng=rng, alpha=curve_alpha, lw=lw)


def make_figure(cells, palettes, base_rng, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Three different looks
    modes = ["bubbles", "flow", "mixed"]
    for ax, palette, mode in zip(axes, palettes, modes):
        # Derive a panel-specific RNG from the base RNG for reproducible variety
        panel_rng = random.Random(base_rng.random())
        draw_panel_curvy(ax, cells, palette, panel_rng, mode=mode)

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
            "Visualize a Python source file as abstract *curvy* shapes.\n"
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
        "--max-cells",
        type=int,
        default=400,
        help="Maximum number of cells/shapes to generate (default: 400)",
    )

    args = parser.parse_args()

    text = read_text(args.source)
    tokens = tokenize_source(text)

    palettes, rng = choose_palettes(text)
    cells = build_cells(
        tokens,
        max_cells=args.max_cells,
        rng=rng,
    )

    title = f"Curvy visualization of: {args.source}"
    fig = make_figure(cells, palettes, rng, title=title)

    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
