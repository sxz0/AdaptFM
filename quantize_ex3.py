#!/usr/bin/env python3
"""
Quantize Qwen3.5-4B to EXL3 format at 4 bpw using ExLlamaV3.

Usage:
    # Basic (4bpw, defaults):
    python3 quantize_ex3.py

    # Custom bitrate:
    python3 quantize_ex3.py --bits 3.5

    # Resume interrupted job:
    python3 quantize_ex3.py --resume

    # Pass any exllamav3 convert.py flags:
    python3 quantize_ex3.py --bits 4.0 --head_bits 6 --cal_rows 512

Requirements:
    pip install exllamav3  (or install from wheel matching your CUDA/torch)
    The wheel for the base Docker image (Python 3.10 + cu128 + torch 2.10.0):
      pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.37/exllamav3-0.0.37+cu128.torch2.10.0-cp310-cp310-linux_x86_64.whl
    For the host venv (Python 3.11):
      pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.37/exllamav3-0.0.37+cu128.torch2.10.0-cp311-cp311-linux_x86_64.whl
"""

import sys
import os
import subprocess
import argparse

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_DIR   = "/data/models/qwen-weights"            # plain FP16 HF weights
OUT_DIR  = "/data/models/qwen-weights-exl3-4bpw"  # EXL3 output
WORK_DIR = "/data/models/qwen-exl3-work"           # working / resume dir


def ensure_exllamav3():
    """Install ExLlamaV3 wheel if not already available."""
    try:
        import exllamav3  # noqa: F401
        print(f"exllamav3 already installed.", flush=True)
        return
    except ImportError:
        pass

    py = sys.version_info
    cuda_tag = "cu128"
    torch_tag = "torch2.10.0"
    cp_tag = f"cp{py.major}{py.minor}"
    wheel = (
        f"https://github.com/turboderp-org/exllamav3/releases/download/v0.0.37/"
        f"exllamav3-0.0.37+{cuda_tag}.{torch_tag}-{cp_tag}-{cp_tag}-linux_x86_64.whl"
    )
    print(f"Installing ExLlamaV3 from:\n  {wheel}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])


def run_quantization(extra_args: list[str]):
    """Run ExLlamaV3 quantization conversion."""
    from exllamav3.conversion.convert_model import parser, main, prepare

    # Build argument list
    base_args = [
        "-i", IN_DIR,
        "-o", OUT_DIR,
        "-w", WORK_DIR,
        "-b", "4.0",        # 4 bits-per-weight
        "--head_bits", "6", # 6 bpw for the output/head layer (better accuracy)
    ]

    # Check if --resume is in extra_args and skip base -i/-o/-w if so
    # (convert.py derives them from -w when resuming)
    all_args = base_args + extra_args

    print("\n" + "="*60, flush=True)
    print(f"ExLlamaV3 quantization: Qwen3.5-4B → 4 bpw EXL3", flush=True)
    print(f"  Input  : {IN_DIR}", flush=True)
    print(f"  Output : {OUT_DIR}", flush=True)
    print(f"  WorkDir: {WORK_DIR}", flush=True)
    print("="*60, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    # Parse and run
    sys.argv = ["convert.py"] + all_args
    _args = parser.parse_args()
    _in_args, _job_state, _ok, _err = prepare(_args)
    if not _ok:
        print(f"\n !! Conversion failed: {_err}", flush=True)
        sys.exit(1)

    main(_in_args, _job_state)
    print(f"\n✅ Quantization complete: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    # Parse only our own known args; pass the rest through to exllamav3
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--skip-install", action="store_true",
                     help="Skip automatic exllamav3 installation check")
    known, extra = pre.parse_known_args()

    if not known.skip_install:
        ensure_exllamav3()

    run_quantization(extra)
