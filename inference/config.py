"""Model registry: per-family pipeline class, default checkpoint and sampling params.

Each `ModelSpec` describes one model *family*. The default repo points at the
open / fast variant (per the project's chosen defaults); pass ``--model-id`` on
the CLI to switch to a dev / larger checkpoint without touching this file.

Notes on the guidance knobs, which differ per family:
  * FLUX.1 / FLUX.2 use a *distilled* ``guidance_scale`` (a conditioning input,
    not classifier-free guidance). schnell is guidance-distilled -> 0.0.
  * Qwen-Image uses real classifier-free guidance via ``true_cfg_scale`` and
    therefore needs a ``negative_prompt`` to have any effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # CLI selector, e.g. "flux1"
    pipeline_class: str            # importable name in `diffusers`, e.g. "FluxPipeline"
    default_repo: str              # HF repo id for the open/fast variant
    # --- sampling defaults (all overridable on the CLI) ---
    steps: int = 28
    guidance_scale: float = 3.5    # distilled guidance (flux) -> passed as guidance_scale
    true_cfg_scale: Optional[float] = None  # set for real CFG models (Qwen)
    supports_negative: bool = False
    default_negative: str = ""
    height: int = 1024
    width: int = 1024
    max_sequence_length: Optional[int] = None
    # Components eligible for quantization. All three families expose "transformer";
    # text encoders are quantized too when present to fit smaller GPUs.
    quantizable_components: tuple[str, ...] = ("transformer",)
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- FLUX.1 -----------------------------------------------------------
    # Open/fast default: schnell (Apache-2.0, 4-step, guidance-distilled).
    # Dev: --model-id black-forest-labs/FLUX.1-dev  (gated; --steps 50 --guidance 3.5)
    "flux1": ModelSpec(
        key="flux1",
        pipeline_class="FluxPipeline",
        default_repo="black-forest-labs/FLUX.1-schnell",
        steps=4,
        guidance_scale=0.0,
        height=1024,
        width=1024,
        max_sequence_length=256,
        quantizable_components=("transformer", "text_encoder_2"),
        notes="schnell: 4 steps, guidance 0.0. For FLUX.1-dev use ~50 steps, guidance 3.5.",
    ),
    # --- FLUX.2 -----------------------------------------------------------
    # Open default: klein (Apache-2.0, distilled). Dev: black-forest-labs/FLUX.2-dev (gated).
    "flux2": ModelSpec(
        key="flux2",
        pipeline_class="Flux2Pipeline",
        default_repo="black-forest-labs/FLUX.2-klein-9B",
        steps=28,
        guidance_scale=4.0,
        height=1024,
        width=1024,
        quantizable_components=("transformer", "text_encoder"),
        notes="Needs recent diffusers. FLUX.2-dev: 50 steps, guidance 2.5 (gated).",
    ),
    # --- Qwen-Image -------------------------------------------------------
    "qwen": ModelSpec(
        key="qwen",
        pipeline_class="QwenImagePipeline",
        default_repo="Qwen/Qwen-Image",
        steps=50,
        guidance_scale=1.0,             # unused; Qwen uses true_cfg_scale
        true_cfg_scale=4.0,
        supports_negative=True,
        default_negative=" ",           # Qwen recommends a single space, not empty
        height=1328,
        width=1328,                     # Qwen's native 1:1 resolution
        quantizable_components=("transformer", "text_encoder"),
        notes="Real CFG via true_cfg_scale; strong text rendering. Native size 1328.",
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        avail = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown model '{key}'. Available: {avail}") from None
