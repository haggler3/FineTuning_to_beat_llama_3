"""Generate docs/architecture.svg and docs/architecture-dark.svg.

Both files come from this one template so the two themes cannot drift apart.
GitHub strips inline SVG from Markdown and does not let `currentColor` reach an
<img>, so the colours are baked in per theme and the README selects between them
with a <picture> element.

Run:  python docs/make_architecture_svg.py
"""

import os

W, H = 1280, 700

LIGHT = {
    "bg": "#ffffff",
    "fg": "#1f2328",
    "muted": "#59636e",
    "border": "#d1d9e0",
    "fill": "#f6f8fa",
    "accent": "#0550ae",
    "accent_fill": "#ddf4ff",
}

DARK = {
    "bg": "#0d1117",
    "fg": "#e6edf3",
    "muted": "#9198a1",
    "border": "#3d444d",
    "fill": "#161b22",
    "accent": "#6cb6ff",
    "accent_fill": "#121d2f",
}


def box(x, y, w, h, title, sub=None, sub2=None, accent=False):
    """A labelled node. Returns SVG markup."""
    stroke = "{accent}" if accent else "{border}"
    fill = "{accent_fill}" if accent else "{fill}"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    ]
    cx = x + w / 2
    if sub2:
        ty = y + h / 2 - 13
    elif sub:
        ty = y + h / 2 - 5
    else:
        ty = y + h / 2 + 5
    parts.append(
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="{{fg}}">{title}</text>'
    )
    if sub:
        parts.append(
            f'<text x="{cx}" y="{ty + 17}" text-anchor="middle" font-size="11" '
            f'fill="{{muted}}">{sub}</text>'
        )
    if sub2:
        parts.append(
            f'<text x="{cx}" y="{ty + 33}" text-anchor="middle" font-size="11" '
            f'fill="{{muted}}">{sub2}</text>'
        )
    return "\n  ".join(parts)


def label(x, y, text, anchor="middle", size=11, color="{muted}", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{text}</text>'
    )


def path(d, accent=False, dashed=False):
    stroke = "{accent}" if accent else "{muted}"
    marker = "arrow-accent" if accent else "arrow"
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    width = "2" if accent else "1.4"
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"'
        f'{dash} marker-end="url(#{marker})"/>'
    )


def build():
    p = []

    # --- title -------------------------------------------------------------
    p.append(label(30, 36, "QLoRA + Dr.ICL pipeline", anchor="start", size=17,
                   color="{fg}", weight="700"))
    p.append(label(30, 58, "One deterministic partition feeds both lanes. "
                           "Demonstrations are retrieved from train, never from the split being scored.",
                   anchor="start", size=12))

    # --- lane headings -----------------------------------------------------
    p.append(label(640, 108, "TRAINING", anchor="start", size=11,
                   color="{muted}", weight="700"))
    p.append(label(880, 364, "EVALUATION", anchor="start", size=11,
                   color="{muted}", weight="700"))

    # --- shared source and splitter ----------------------------------------
    p.append(box(30, 300, 150, 56, "HF dataset", "ChatDoctor / GSM8K"))
    p.append(box(225, 286, 180, 84, "build_splits()", "seed · max_samples",
                 "→ split fingerprint"))

    # --- the three splits --------------------------------------------------
    p.append(box(455, 140, 110, 40, "train 80%"))
    p.append(box(455, 306, 110, 40, "val 10%"))
    p.append(box(455, 500, 110, 40, "test 10%"))

    # --- training lane -----------------------------------------------------
    p.append(box(640, 124, 180, 72, "tokenize",
                 "completion-only labels", "pad + prompt → −100"))
    p.append(box(880, 124, 180, 72, "4-bit NF4 base",
                 "+ LoRA adapters", "r=8, ~0.2% trainable"))
    p.append(box(1120, 132, 130, 56, "LoRA adapter", "a few MB"))

    # --- evaluation lane ---------------------------------------------------
    p.append(box(640, 380, 180, 64, "GTR-T5 retriever", "top-k demonstrations",
                 accent=True))
    p.append(box(640, 486, 180, 64, "ICL prompt", "k demos + question"))
    p.append(box(880, 486, 180, 64, "base + adapter", "merged for inference"))
    p.append(box(1120, 486, 130, 64, "BERTScore", "judge model"))

    # --- edges -------------------------------------------------------------
    p.append(path("M 180 328 H 219"))                       # dataset -> splitter
    p.append(path("M 405 312 H 425 V 160 H 449"))           # splitter -> train
    p.append(path("M 405 328 H 449"))                       # splitter -> val
    p.append(path("M 405 344 H 425 V 520 H 449"))           # splitter -> test

    p.append(path("M 565 160 H 634"))                       # train -> tokenize
    p.append(path("M 820 160 H 874"))                       # tokenize -> model
    p.append(path("M 1060 160 H 1114"))                     # model -> adapter

    # val feeds the trainer's periodic evaluation
    p.append(path("M 510 306 V 230 H 970 V 190"))
    p.append(label(800, 222, "eval loss"))

    # test feeds the prompt builder
    p.append(path("M 565 520 H 634"))

    # THE edge the whole repair turns on: demos come from train, not from test.
    # Routed out to x=600 so it clears the val node rather than crossing it.
    p.append(path("M 530 180 V 250 H 600 V 412 H 634", accent=True))
    p.append(label(604, 243, "retrieval corpus", anchor="start", size=11,
                   color="{accent}", weight="600"))
    p.append(label(604, 258, "(train only, never test)", anchor="start", size=11,
                   color="{accent}"))

    # guard pill sitting on that edge
    p.append(
        '<rect x="516" y="348" width="168" height="26" rx="13" '
        'fill="{accent_fill}" stroke="{accent}" stroke-width="1.5"/>'
    )
    p.append(label(600, 365, "assert_no_leakage()", size=11,
                   color="{accent}", weight="600"))

    p.append(path("M 730 444 V 480", accent=True))          # retriever -> prompt
    p.append(path("M 820 518 H 874"))                       # prompt -> model
    p.append(path("M 1060 518 H 1114"))                     # model -> judge
    p.append(label(1090, 468, "continuation", size=10))
    p.append(label(1090, 480, "only", size=10))

    # --- fingerprint contract note ----------------------------------------
    p.append(
        '<rect x="30" y="576" width="372" height="92" rx="6" '
        'fill="none" stroke="{border}" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    p.append(label(46, 600, "Split fingerprint", anchor="start", size=12,
                   color="{fg}", weight="600"))
    p.append(label(46, 620, "Both entry points print it. If the two runs report",
                   anchor="start", size=11))
    p.append(label(46, 636, "different fingerprints they are not talking about",
                   anchor="start", size=11))
    p.append(label(46, 652, "the same held-out set.", anchor="start", size=11))
    p.append(path("M 216 576 V 420 H 315 V 378", dashed=True))

    return "\n  ".join(p)


TEMPLATE = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Architecture of the QLoRA plus Dr.ICL pipeline. A single build_splits call partitions the dataset into train, validation and test, and emits a fingerprint that both entry points print. The training lane tokenizes the train split with completion-only labels, attaches LoRA adapters to a 4-bit quantized base model, and emits an adapter. The evaluation lane retrieves demonstrations from the train split only, guarded by an assert_no_leakage check, builds an in-context prompt around the test split, generates from the merged base plus adapter, and scores the continuation alone with a BERTScore judge." font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{muted}"/>
    </marker>
    <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{accent}"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="{bg}"/>
  {body}
</svg>
'''


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    body = build()

    for name, palette in [("architecture.svg", LIGHT),
                          ("architecture-dark.svg", DARK)]:
        # Resolve the colour placeholders in the body first, then drop the
        # finished body into the outer template.
        svg = TEMPLATE.format(W=W, H=H, body=body.format(**palette), **palette)
        out = os.path.join(here, name)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        print("wrote", out)


if __name__ == "__main__":
    main()
