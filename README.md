# DAC+Q4-ITS (Minimal Reproducible Starter)

This starter runs an **INT4 + low-rank adapter compensation** pipeline on a **toy transformer** to prove the flow:
corpus → export → quantize (INT4) → eigenspace (SVD) → build adapters (U,D) → inject → inference → evaluation.

> Swap `ToyTransformer` with your LLaMA-derived model and wire TRT for Jetson when ready.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
python scripts/01_prepare_calibration.py
python scripts/02_export_onnx.py
python scripts/03_quantize_int4.py
python scripts/04_compute_eigenspaces.py
python scripts/05_inject_adapters.py
python scripts/07_run_inference.py --prompt "avoid tolls and reach airport by 6pm"
python scripts/08_eval_nln.py
```

## What this starter gives
- Working INT4 quantization (per-row scales)
- Rank-8 adapters computed from calibration activations
- Compressed forward pass = dequantized(INT4 matmul) + U(Dx)
- Simple EM / SOFT-F1 metrics demo

## Swap in real model
- Replace `src/dac_q4_its/modeling/loader.py` to load your HF/LLaMA weights.
- Populate `scripts/02_export_onnx.py` to export real projections.
- Replace toy heads with your real Q/K/V/FFN matrices; rest of pipeline stays same.
