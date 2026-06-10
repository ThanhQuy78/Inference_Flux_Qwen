# FLUX.1 / FLUX.2 / Qwen-Image inference (diffusers)

Unified text-to-image inference for three model families behind one CLI:

| family  | default (open/fast) checkpoint        | pipeline           | guidance        |
|---------|----------------------------------------|--------------------|-----------------|
| `flux1` | `black-forest-labs/FLUX.1-schnell`     | `FluxPipeline`     | distilled (0.0) |
| `flux2` | `black-forest-labs/FLUX.2-klein-9B`    | `Flux2Pipeline`    | distilled (~4)  |
| `qwen`  | `Qwen/Qwen-Image`                      | `QwenImagePipeline`| true CFG (~4)   |

Switch to a gated **dev** checkpoint with `--model-id` + `--hf-token`.

## Install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# Install a CUDA build of torch first (pick your CUDA version):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# FLUX.2 / Qwen-Image are new — if their pipeline import fails, get latest diffusers:
pip install -U "git+https://github.com/huggingface/diffusers"
```

## Usage

```powershell
# FLUX.1 schnell (4-step, open)
python generate.py flux1 -p "a fox in a snowy forest, cinematic lighting"

# FLUX.2 klein, 4-bit quantized
python generate.py flux2 -p "an astronaut riding a horse" --quantize 4bit

# Qwen-Image with negative prompt, size, seed
python generate.py qwen -p "poster reading 'HELLO WORLD'" `
    --negative "blurry, low quality" --size 1328x1328 --seed 42

# Gated dev checkpoint
python generate.py flux1 --model-id black-forest-labs/FLUX.1-dev `
    --steps 50 --guidance 3.5 -p "..." --hf-token $env:HF_TOKEN
```

Output PNGs are written to `outputs/` (override with `-o`).

## Fitting it on your GPU

These are large (FLUX.1 ≈ 12B, FLUX.2-klein ≈ 9B, Qwen-Image ≈ 20B). To reduce VRAM:

| flag                    | effect                                              |
|-------------------------|-----------------------------------------------------|
| `--quantize 4bit`       | bitsandbytes NF4 — biggest VRAM cut (CUDA only)     |
| `--offload model`       | move modules to GPU only while in use               |
| `--offload sequential`  | finest-grained offload, lowest VRAM, slowest        |
| `--vae-tiling`          | tile VAE decode for high resolutions                |

Combine `--quantize 4bit --offload model` to run the big models on ~8–12 GB cards.

> **This machine:** torch here is a CPU-only build and the GPU has 4 GB VRAM —
> too little for these models on-GPU. Run on a CUDA GPU with ≥8 GB (use the
> flags above), or expect very slow CPU execution. `--quantize` requires CUDA.

## Library use

```python
from inference import build_pipeline, GenerationConfig, generate

pipe = build_pipeline("qwen", quantize="4bit", offload="model")
imgs = generate(pipe, GenerationConfig(prompt="a teapot", seed=0))
imgs[0].save("teapot.png")
```

## Layout

```
generate.py            CLI entry point
inference/config.py    per-family ModelSpec registry (checkpoints, defaults)
inference/factory.py   pipeline construction + generation (device/quant/offload)
```
