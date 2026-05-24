#!/usr/bin/env python3
"""
INT4 GPTQ quantization of the pruned Qwen3.5 model using gptqmodel.
Produces the same format as qwen-weights-gptq but from the 28-layer pruned model.

Usage:
    .venv/bin/python quantize_pruned.py
"""
from gptqmodel import GPTQModel, QuantizeConfig

INPUT_PATH  = "qwen-weights-pruned"
OUTPUT_PATH = "qwen-weights-pruned-gptq"

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
    "If a train travels at 80 km/h for 2.5 hours, how far? Answer: 200 km.",
    "Describe three key properties of a good cryptographic hash function.",
    "A protein complex dissociates at high salt: stabilized by (C) Electrostatic interactions.",
    "Rewrite using passive voice: 'The scientist conducted the experiment.'",
    "What is the significance of the Pauli exclusion principle?",
    "Describe the water cycle and its importance to Earth's ecosystem.",
]

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,   # required for vLLM Marlin kernel
    sym=True,
)

print(f"Loading {INPUT_PATH}...", flush=True)
model = GPTQModel.load(INPUT_PATH, quant_config)

print("Quantizing (INT4 GPTQ)...", flush=True)
model.quantize(CALIB_DATA)

print(f"Saving to {OUTPUT_PATH}...", flush=True)
model.save(OUTPUT_PATH)
print("Done.", flush=True)
