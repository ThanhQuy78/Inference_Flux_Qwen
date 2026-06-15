#!/usr/bin/env python
"""Side-by-side comparison of FLUX.1 / FLUX.2 / Qwen-Image on identical inputs.

Runs the SAME prompts through each model with the SAME seed and output size,
then builds a labeled contact sheet (rows = prompts, columns = models) so the
differences are easy to eyeball.

Examples
--------
  # Compare all three on two prompts, fixed seed + size
  python compare.py --models flux1 flux2 qwen --seed 42 --size 1024x1024 \
      -p "a fox in a snowy forest, cinematic" \
      -p "a poster that says 'HELLO WORLD'"

  # Prompts from a file (one per line), 4-bit + offload to fit low VRAM
  python compare.py --prompts-file prompts.txt --quantize 4bit --offload model

Fairness notes
--------------
* A shared --seed makes each model deterministic, but it does NOT give the
  models the "same noise" (their latent shapes/schedulers differ). It only
  makes the comparison reproducible.
* By default each model uses its OWN recommended step count (schnell needs 4,
  the others ~28-50). Use --equal-steps N to force an identical budget — fair
  on compute, but unfavourable to models tuned for a specific schedule.
* All models render at a single shared --size so the grid lines up, even if a
  model has a different native resolution (e.g. Qwen's 1328).
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

from inference.config import MODEL_REGISTRY
from inference.factory import GenerationConfig, build_pipeline, generate


def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"--size must be WxH, got {value!r}")


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n] or "prompt").rstrip("-")


def free_pipeline(pipe) -> None:
    """Drop a pipeline and release GPU memory before loading the next model."""
    try:
        import torch

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_prompts(args) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise SystemExit("No prompts given. Use -p/--prompt or --prompts-file.")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-p", "--prompt", action="append", help="a prompt (repeatable)")
    p.add_argument("--prompts-file", help="text file, one prompt per line (# = comment)")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                   choices=list(MODEL_REGISTRY), help="models to compare")
    p.add_argument("--seed", type=int, default=0, help="shared seed (default 0)")
    p.add_argument("--size", type=parse_size, default=(1024, 1024), help="shared WxH")
    p.add_argument("--equal-steps", type=int, default=None,
                   help="force identical step count for every model")
    p.add_argument("--negative", default=None, help="negative prompt (Qwen / dev variants)")

    # runtime (forwarded to build_pipeline)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--quantize", default="none", choices=["none", "4bit", "8bit"])
    p.add_argument("--offload", default="none", choices=["none", "model", "sequential"])
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))

    p.add_argument("-o", "--out-dir", default="outputs/compare", help="output directory")
    p.add_argument("--thumb", type=int, default=512, help="grid cell size in px")
    p.add_argument("--no-grid", action="store_true", help="skip the contact sheet")
    return p


def make_grid(results: dict, models: list[str], prompts: list[str],
              out_path: Path, thumb: int) -> None:
    """Compose a labeled grid: header row of model names, left column of prompts."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_big = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_big = font

    pad = 10
    label_w = 260            # left column width for prompt text
    header_h = 40            # top row height for model names
    cell = thumb
    cols, rows = len(models), len(prompts)

    grid_w = label_w + cols * (cell + pad) + pad
    grid_h = header_h + rows * (cell + pad) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Column headers (model names).
    for c, model in enumerate(models):
        x = label_w + c * (cell + pad) + pad
        draw.text((x, 10), model, fill="black", font=font_big)

    def wrap(text: str, width_px: int) -> str:
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if draw.textlength(trial, font=font) <= width_px:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines[:6])

    for r, prompt in enumerate(prompts):
        y = header_h + r * (cell + pad) + pad
        draw.multiline_text((pad, y), wrap(prompt, label_w - 2 * pad),
                            fill="black", font=font, spacing=2)
        for c, model in enumerate(models):
            x = label_w + c * (cell + pad) + pad
            entry = results.get((model, r))
            box = (x, y, x + cell, y + cell)
            if entry and Path(entry).exists():
                img = Image.open(entry).convert("RGB")
                img.thumbnail((cell, cell))
                ox = x + (cell - img.width) // 2
                oy = y + (cell - img.height) // 2
                canvas.paste(img, (ox, oy))
            else:
                draw.rectangle(box, outline="red")
                draw.text((x + 8, y + 8), "(failed)", fill="red", font=font)
            draw.rectangle(box, outline="lightgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[grid] {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_prompts(args)
    width, height = args.size
    out_dir = Path(args.out_dir)

    print(f"Comparing {args.models} on {len(prompts)} prompt(s), "
          f"seed={args.seed}, size={width}x{height}")

    results: dict[tuple[str, int], str] = {}
    timings: dict[str, float] = {}

    # Outer loop = model (load once, run every prompt, then free VRAM).
    for model in args.models:
        print(f"\n=== {model} ===")
        t0 = time.time()
        try:
            pipe = build_pipeline(
                model,
                device=args.device,
                dtype=args.dtype,
                quantize=args.quantize,
                offload=args.offload,
                hf_token=args.hf_token,
            )
        except Exception as e:  # missing weights / OOM / old diffusers -> skip model
            print(f"[skip] could not load {model}: {e}")
            continue

        for i, prompt in enumerate(prompts):
            try:
                imgs = generate(pipe, GenerationConfig(
                    prompt=prompt,
                    negative_prompt=args.negative,
                    steps=args.equal_steps,
                    height=height, width=width,
                    seed=args.seed,
                ))
                path = out_dir / model / f"{i:02d}_{slugify(prompt)}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                imgs[0].save(path)
                results[(model, i)] = str(path)
                print(f"  [{i}] {path}")
            except Exception as e:
                print(f"  [{i}] FAILED: {e}")

        timings[model] = time.time() - t0
        free_pipeline(pipe)

    if timings:
        print("\nLoad+gen wall time per model:")
        for m, dt in timings.items():
            print(f"  {m:7s} {dt:6.1f}s")

    if not args.no_grid and results:
        make_grid(results, args.models, prompts, out_dir / "grid.png", args.thumb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
