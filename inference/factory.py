"""Build a diffusers pipeline for a model family and run generation.

Handles the cross-cutting concerns that are the same for every family:
device/dtype selection, optional bitsandbytes quantization, CPU offloading,
VAE slicing/tiling, and assembling the right call kwargs (distilled
guidance vs. real CFG, negative prompts, sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .config import ModelSpec, get_spec


# --------------------------------------------------------------------------- #
# Device / dtype helpers
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[requested]
    # bf16 is the native dtype for all three families on CUDA; CPU stays fp32.
    return torch.bfloat16 if device == "cuda" else torch.float32


def _load_pipeline(repo: str, spec: ModelSpec, load_kwargs: dict):
    """Load a pipeline, letting diffusers pick the exact class from the repo's
    model_index.json (`_class_name`).

    This is more robust than hardcoding a class per family: e.g. FLUX.2 [dev]
    declares `Flux2Pipeline` (Mistral-3 text encoder) while FLUX.2 [klein]
    declares `Flux2KleinPipeline` (Qwen3 text encoder). Hardcoding one breaks
    the other; `DiffusionPipeline.from_pretrained` reads the right one.
    """
    from diffusers import DiffusionPipeline

    try:
        return DiffusionPipeline.from_pretrained(repo, **load_kwargs)
    except (ValueError, ImportError, AttributeError, OSError) as e:
        raise ImportError(
            f"Could not load '{repo}'. Your installed diffusers may be too old for "
            f"this checkpoint (it expects a pipeline like '{spec.pipeline_class}'). "
            f"Upgrade and retry:\n"
            f"  pip install -U 'git+https://github.com/huggingface/diffusers'\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def _build_quant_config(spec: ModelSpec, mode: str, compute_dtype: torch.dtype):
    """Return a diffusers PipelineQuantizationConfig, or None.

    `mode` is one of {"none", "4bit", "8bit"}. Quantization is bitsandbytes-based
    and requires a CUDA GPU + the `bitsandbytes` package.
    """
    if mode == "none":
        return None
    from diffusers import PipelineQuantizationConfig  # lazy: only when requested

    if mode == "4bit":
        backend = "bitsandbytes_4bit"
        kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": compute_dtype,
        }
    elif mode == "8bit":
        backend = "bitsandbytes_8bit"
        kwargs = {"load_in_8bit": True}
    else:
        raise ValueError(f"Unknown quantize mode: {mode}")

    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs=kwargs,
        components_to_quantize=list(spec.quantizable_components),
    )


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_pipeline(
    model: str,
    *,
    model_id: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    quantize: str = "none",          # none | 4bit | 8bit
    offload: str = "none",           # none | model | sequential
    vae_slicing: bool = True,
    vae_tiling: bool = False,
    hf_token: Optional[str] = None,
):
    spec = get_spec(model)
    repo = model_id or spec.default_repo
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)

    if quantize != "none" and device != "cuda":
        raise RuntimeError(
            f"--quantize {quantize} requires a CUDA GPU (bitsandbytes). "
            f"Resolved device is '{device}'."
        )

    load_kwargs: dict = {"torch_dtype": torch_dtype}
    if hf_token:
        load_kwargs["token"] = hf_token
    quant_cfg = _build_quant_config(spec, quantize, torch_dtype)
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg

    # bitsandbytes-quantized weights are incompatible with sequential CPU offload:
    # accelerate tries to rebuild Params4bit/Int8Params on the `meta` device and the
    # bnb/accelerate versions disagree (TypeError: unexpected '_is_hf_initialized').
    # Model-level offload works fine and gives nearly the same VRAM savings.
    if quant_cfg is not None and offload == "sequential":
        print("[warn] sequential offload is incompatible with bitsandbytes quantization; "
              "falling back to --offload model.")
        offload = "model"

    print(f"[load] {spec.pipeline_class} <- {repo}")
    print(f"[load] device={device} dtype={torch_dtype} quantize={quantize} offload={offload}")
    pipe = _load_pipeline(repo, spec, load_kwargs)

    # Placement. Offloading and an explicit .to(device) are mutually exclusive;
    # offloading manages device placement itself and must NOT be combined with .to().
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "none":
        pipe.to(device)
    else:
        raise ValueError(f"Unknown offload mode: {offload}")

    # VAE memory savings (cheap, big help at high resolutions / low VRAM).
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    pipe._zen_spec = spec  # stash for generate()
    pipe._zen_device = device
    return pipe


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    seed: Optional[int] = None
    num_images: int = 1


def _build_call_kwargs(spec: ModelSpec, cfg: GenerationConfig, device: str) -> dict:
    kwargs: dict = {
        "prompt": cfg.prompt,
        "num_inference_steps": cfg.steps if cfg.steps is not None else spec.steps,
        "height": cfg.height if cfg.height is not None else spec.height,
        "width": cfg.width if cfg.width is not None else spec.width,
        "num_images_per_prompt": cfg.num_images,
    }

    if spec.true_cfg_scale is not None:
        # Real CFG model (Qwen): true_cfg_scale + negative_prompt.
        kwargs["true_cfg_scale"] = (
            cfg.true_cfg_scale if cfg.true_cfg_scale is not None else spec.true_cfg_scale
        )
        neg = cfg.negative_prompt
        if neg is None:
            neg = spec.default_negative
        kwargs["negative_prompt"] = neg
    else:
        # Distilled-guidance model (FLUX). No CFG / negative prompt.
        kwargs["guidance_scale"] = (
            cfg.guidance_scale if cfg.guidance_scale is not None else spec.guidance_scale
        )
        if spec.supports_negative and cfg.negative_prompt:
            kwargs["negative_prompt"] = cfg.negative_prompt

    if spec.max_sequence_length is not None:
        kwargs["max_sequence_length"] = spec.max_sequence_length

    if cfg.seed is not None:
        # Generator must live on a real device; CPU is safe and reproducible
        # even when the pipeline is offloaded.
        gen_device = "cpu" if device in ("cpu", "mps") else device
        kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(cfg.seed)

    return kwargs


def generate(pipe, cfg: GenerationConfig):
    spec: ModelSpec = pipe._zen_spec
    device: str = pipe._zen_device
    call_kwargs = _build_call_kwargs(spec, cfg, device)

    printable = {k: v for k, v in call_kwargs.items() if k != "generator"}
    print(f"[generate] {printable}")
    result = pipe(**call_kwargs)
    return result.images


def save_images(images, out_dir: str, prefix: str) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = out / (f"{prefix}.png" if len(images) == 1 else f"{prefix}_{i}.png")
        img.save(p)
        paths.append(str(p))
    return paths
