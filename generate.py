#!/usr/bin/env python
"""CLI for FLUX.1 / FLUX.2 / Qwen-Image text-to-image inference.

Examples
--------
  # FLUX.1 schnell (open, 4-step) on a CUDA GPU
  python generate.py flux1 -p "a fox in a snowy forest, cinematic"

  # FLUX.2 klein, 4-bit quantized to fit a smaller GPU
  python generate.py flux2 -p "an astronaut riding a horse" --quantize 4bit

  # Qwen-Image with a negative prompt and explicit size/seed
  python generate.py qwen -p "a poster that says 'HELLO'" \
      --negative "blurry, low quality" --size 1328x1328 --seed 42

  # Switch to a gated dev checkpoint + token
  python generate.py flux1 --model-id black-forest-labs/FLUX.1-dev \
      --steps 50 --guidance 3.5 -p "..." --hf-token $env:HF_TOKEN

Run with low VRAM:  add `--offload model` (or `--offload sequential` for the
tightest fit) and/or `--quantize 4bit`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from Inference_Flux_Qwen.inference.config import MODEL_REGISTRY
from Inference_Flux_Qwen.inference.factory import (
    GenerationConfig,
    build_pipeline,
    generate,
    save_images,
)


def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"--size must be WxH (e.g. 1024x1024), got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Text-to-image inference for FLUX.1, FLUX.2 and Qwen-Image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("model", choices=list(MODEL_REGISTRY), help="model family")
    p.add_argument("-p", "--prompt", required=True, help="text prompt")
    p.add_argument("-n", "--negative", default=None, help="negative prompt (Qwen / dev only)")

    # Sampling overrides (default to the family's spec when omitted).
    p.add_argument("--steps", type=int, default=None, help="num inference steps")
    p.add_argument("--guidance", type=float, default=None, help="distilled guidance_scale (FLUX)")
    p.add_argument("--true-cfg", type=float, default=None, help="true_cfg_scale (Qwen)")
    p.add_argument("--size", type=parse_size, default=None, help="image size WxH, e.g. 1024x1024")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--num-images", type=int, default=1, help="images per prompt")

    # Model / runtime.
    p.add_argument("--model-id", default=None, help="override HF repo id (e.g. a dev checkpoint)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--quantize", default="none", choices=["none", "4bit", "8bit"],
                   help="bitsandbytes quantization (CUDA only)")
    p.add_argument("--offload", default="none", choices=["none", "model", "sequential"],
                   help="CPU offload strategy for low VRAM")
    p.add_argument("--no-vae-slicing", action="store_true", help="disable VAE slicing")
    p.add_argument("--vae-tiling", action="store_true", help="enable VAE tiling (very low VRAM)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF token for gated repos (or set HF_TOKEN)")

    # Output.
    p.add_argument("-o", "--out-dir", default="outputs", help="output directory")
    p.add_argument("--name", default=None, help="output filename prefix")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    width = height = None
    if args.size is not None:
        width, height = args.size

    pipe = build_pipeline(
        args.model,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        quantize=args.quantize,
        offload=args.offload,
        vae_slicing=not args.no_vae_slicing,
        vae_tiling=args.vae_tiling,
        hf_token=args.hf_token,
    )

    cfg = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative,
        steps=args.steps,
        guidance_scale=args.guidance,
        true_cfg_scale=args.true_cfg,
        height=height,
        width=width,
        seed=args.seed,
        num_images=args.num_images,
    )

    t0 = time.time()
    images = generate(pipe, cfg)
    dt = time.time() - t0

    prefix = args.name or f"{args.model}_{args.seed if args.seed is not None else 'rand'}"
    paths = save_images(images, args.out_dir, prefix)
    print(f"[done] {len(paths)} image(s) in {dt:.1f}s:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
