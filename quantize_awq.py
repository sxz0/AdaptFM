#!/usr/bin/env python3
"""
GPTQ 4-bit quantization of Qwen3.5-4B using GPTQModel.
Produces weights loadable by vLLM with --quantization gptq_marlin.

Usage:
    python quantize_awq.py [--model PATH] [--output PATH]

Output: /data/models/qwen-weights-gptq
"""
import argparse

DEFAULT_INPUT  = "/data/models/qwen-weights"
DEFAULT_OUTPUT = "/data/models/qwen-weights-gptq"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print(f"Input : {args.model}", flush=True)
    print(f"Output: {args.output}", flush=True)

    quant_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,   # standard; vLLM Marlin kernel expects 128
        desc_act=False,   # False = faster inference (actorder=False)
    )

    print("[1/3] Loading tokenizer + model in FP16 …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=quant_config,
    )

    print("[2/3] Loading calibration data (wikitext2, 128 samples) …", flush=True)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s for s in data["text"] if len(s.strip()) > 100][:128]
    samples = [tokenizer(t, return_tensors="pt", truncation=True, max_length=512) for t in texts]

    print("[3/3] Quantizing …", flush=True)
    model.quantize(samples)

    print(f"Saving to {args.output} …", flush=True)
    model.save_quantized(args.output, use_safetensors=True)
    tokenizer.save_pretrained(args.output)

    print("\n✅ GPTQ quantization complete!", flush=True)
    print(f"   Load in vLLM:  --model {args.output} --quantization gptq_marlin", flush=True)


if __name__ == "__main__":
    main()
