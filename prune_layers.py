#!/usr/bin/env python3
"""
Layer pruning for Qwen3.5-VL using Block Influence (BI) scoring.

Architecture: Qwen3_5ForConditionalGeneration
  - Language layers at: model.language_model.layers.{i}.*
  - Visual encoder at:  model.visual.* (not pruned)
  - Config key:         text_config.num_hidden_layers

Strategy:
  1. Compute BI scores using INT4 GPTQ model (3.8 GB, fits in RAM).
     Hooks attach to gptq_model.model.language_model.layers[i].
  2. Prune FP16 weights directly via safetensors (no full model load → no OOM).
     Process each shard, skip pruned layers, remap indices.
  3. Update config.json and index.json, copy tokenizer files.

Usage:
    .venv/bin/python prune_layers.py [--prune N]
    Default: prune 4 layers (32 → 28).
"""
import argparse
import gc
import glob
import json
import os
import re
import shutil

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

GPTQ_PATH   = "qwen-weights-gptq"
FP16_PATH   = "qwen-weights"
OUTPUT_PATH = "qwen-weights-pruned"

CALIB_DATA = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the concept of quantum entanglement in simple terms.",
    "What is the capital of France? Answer: Paris.",
    "Solve the equation: 2x + 5 = 13. Step: subtract 5, divide by 2, x=4.",
    "Mitochondria: (A) Protein synthesis (B) Energy production. Answer: B",
    "Write a Python function that returns the nth Fibonacci number recursively.",
    "General relativity, published by Einstein in 1915, describes gravity as spacetime curvature.",
    "Summarize the main causes of World War I in three sentences.",
    "Key differences between supervised and unsupervised machine learning.",
    "Calculate the derivative of f(x) = 3x^3 - 2x^2 + x - 5.",
    "Describe the process of DNA replication in biology.",
    "Three advantages and disadvantages of renewable energy sources.",
    "Difference between a stack and a queue data structure.",
    "Significance of the Magna Carta in constitutional law.",
    "A protein complex dissociates at high salt: stabilized by (C) Electrostatic interactions.",
]

LAYER_RE = re.compile(r"(model\.language_model\.layers\.)(\d+)(\..*)")


def compute_bi_scores(num_layers):
    """Load INT4 GPTQ model, compute BI scores, unload."""
    from gptqmodel import GPTQModel

    print(f"Loading GPTQ model for scoring ({GPTQ_PATH})...", flush=True)
    m = GPTQModel.from_quantized(GPTQ_PATH, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(GPTQ_PATH, local_files_only=True)

    layers = m.model.model.language_model.layers
    assert len(layers) == num_layers, f"Expected {num_layers} layers, got {len(layers)}"

    scores = torch.zeros(num_layers)
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            x_in  = inp[0].detach().float()
            x_out = (out[0] if isinstance(out, tuple) else out).detach().float()
            bi = 1.0 - F.cosine_similarity(
                x_in.reshape(-1, x_in.shape[-1]),
                x_out.reshape(-1, x_out.shape[-1]),
                dim=-1,
            ).mean().item()
            scores[idx] += bi
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    m.eval()
    with torch.no_grad():
        for text in CALIB_DATA:
            ids = tokenizer(text, return_tensors="pt",
                            max_length=128, truncation=True)
            try:
                m.model(**ids)
            except Exception:
                # Some calib samples may fail with multimodal model — skip
                pass

    for h in handles:
        h.remove()

    del m
    gc.collect()
    return scores / max(1, len(CALIB_DATA))


def prune_safetensors(fp16_path, output_path, prune_idx, keep_idx):
    """Remap safetensors shards: drop pruned layers, renumber kept layers."""
    os.makedirs(output_path, exist_ok=True)

    index_path = os.path.join(fp16_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # Group keys by shard
        shard_to_keys: dict[str, list] = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
    else:
        # Single-file model
        shard_name = "model.safetensors"
        shard_to_keys = {shard_name: None}
        weight_map = None

    new_weight_map: dict[str, str] = {}
    out_shard_idx = 1

    for shard_name, keys in shard_to_keys.items():
        shard_path = os.path.join(fp16_path, shard_name)
        print(f"  Processing {shard_name}...", flush=True)
        tensors = load_file(shard_path)

        new_tensors: dict[str, torch.Tensor] = {}
        for key, val in tensors.items():
            hit = LAYER_RE.match(key)
            if hit:
                idx = int(hit.group(2))
                if idx in prune_idx:
                    continue  # drop this layer
                new_idx = keep_idx.index(idx)
                new_key = f"{hit.group(1)}{new_idx}{hit.group(3)}"
            else:
                new_key = key

            new_tensors[new_key] = val
            if weight_map is not None:
                new_weight_map[new_key] = f"model-{out_shard_idx:05d}-of-{len(shard_to_keys):05d}.safetensors"

        out_name = f"model-{out_shard_idx:05d}-of-{len(shard_to_keys):05d}.safetensors"
        save_file(new_tensors, os.path.join(output_path, out_name))
        print(f"    Saved {out_name} ({len(new_tensors)} tensors)", flush=True)
        out_shard_idx += 1
        del tensors, new_tensors
        gc.collect()

    # Write new index
    if weight_map is not None:
        total_size = index.get("metadata", {}).get("total_size", 0)
        new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f, indent=2)


def update_config(fp16_path, output_path, keep_idx):
    with open(os.path.join(fp16_path, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    old_n = tc["num_hidden_layers"]
    tc["num_hidden_layers"] = len(keep_idx)
    # Prune all per-layer list fields (layer_types, etc.)
    for k, v in tc.items():
        if isinstance(v, list) and len(v) == old_n:
            tc[k] = [v[i] for i in keep_idx]
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", type=int, default=4,
                        help="Number of layers to remove (default 4, 32→28)")
    args = parser.parse_args()

    # Read num_hidden_layers from config
    with open(os.path.join(FP16_PATH, "config.json")) as f:
        cfg = json.load(f)
    num_layers = cfg["text_config"]["num_hidden_layers"]
    print(f"Model has {num_layers} language layers. Pruning {args.prune}.", flush=True)

    # ── Step 1: BI scores ─────────────────────────────────────────────────────
    scores = compute_bi_scores(num_layers)

    print("\nBI scores per layer:")
    for i, s in enumerate(scores.tolist()):
        print(f"  Layer {i:2d}: {s:.4f}")

    prune_idx = set(scores.argsort()[:args.prune].tolist())
    keep_idx  = sorted(set(range(num_layers)) - prune_idx)
    print(f"\nPruning layers : {sorted(prune_idx)}")
    print(f"Keeping layers : {keep_idx}")

    # ── Step 2: Prune safetensors shards ─────────────────────────────────────
    print(f"\nPruning FP16 weights from {FP16_PATH}...", flush=True)
    prune_safetensors(FP16_PATH, OUTPUT_PATH, prune_idx, keep_idx)

    # ── Step 3: Update config + copy tokenizer files ─────────────────────────
    update_config(FP16_PATH, OUTPUT_PATH, keep_idx)
    for fname in os.listdir(FP16_PATH):
        if fname.endswith((".json", ".jinja", ".model", ".tiktoken")) \
                and fname not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(FP16_PATH, fname),
                        os.path.join(OUTPUT_PATH, fname))

    print(f"\nDone. {num_layers} → {len(keep_idx)} layers saved to {OUTPUT_PATH}.", flush=True)


if __name__ == "__main__":
    main()
