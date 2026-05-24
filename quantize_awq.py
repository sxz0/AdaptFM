#!/usr/bin/env python3
"""
True AWQ W4A16 quantization using autoawq.
Produces weights loadable by vLLM with --quantization awq_marlin.

Usage:
    python quantize_awq.py [--model PATH] [--output PATH]
    python quantize_awq.py --model qwen-weights-pruned --output qwen-weights-pruned-awq

Requires: pip install autoawq
Needs GPU with enough VRAM to load the FP16 model (A10G works, GTX 1650 does not).
"""
import argparse

DEFAULT_INPUT  = "qwen-weights-pruned"
DEFAULT_OUTPUT = "qwen-weights-pruned-awq"

CALIB_DATA = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the concept of quantum entanglement in simple terms.",
    "What is the capital of France? Answer: Paris.",
    "Solve the equation: 2x + 5 = 13. Subtract 5, divide by 2, x=4.",
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
    "If a train travels at 80 km/h for 2.5 hours, how far does it travel? Answer: 200 km.",
    "Describe three key properties of a good cryptographic hash function.",
    "A protein complex dissociates at high salt: stabilized by (C) Electrostatic interactions.",
    "Rewrite using passive voice: 'The scientist conducted the experiment.'",
    "What is the significance of the Pauli exclusion principle in quantum mechanics?",
    "Describe the water cycle and its importance to Earth's ecosystem.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model from {args.model} in FP16...", flush=True)
    model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "zero_point": True,   # asymmetric (standard AWQ)
        "q_group_size": 128,  # matches gptq_marlin group_size=128
        "w_bit": 4,           # INT4
        "version": "GEMM",    # use GEMM; vLLM will select Marlin kernel at runtime
    }

    print("Quantizing with AWQ (this takes ~15 min on A10G)...", flush=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=CALIB_DATA)

    print(f"Saving to {args.output}...", flush=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)

    import shutil, os
    for fname in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(args.model, fname)
        dst = os.path.join(args.output, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {fname}", flush=True)

    print("Done.", flush=True)
    print(f"Load in vLLM with: --quantization awq_marlin", flush=True)


if __name__ == "__main__":
    main()
