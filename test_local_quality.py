#!/usr/bin/env python3
"""
Quick quality sanity check for the pruned GPTQ model on CPU.
Tests a handful of IFEval-style instruction-following prompts.
Usage: .venv/bin/python3 test_local_quality.py [model_path]
"""
import sys, re, time

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "qwen-weights-pruned-gptq"

PROMPTS = [
    # IFEval-style: must follow a specific format or constraint
    ("Write exactly 3 bullet points about Python. Each bullet must start with '- '.",
     lambda r: r.count("\n- ") >= 2 or r.startswith("- "),
     "3 bullet points"),
    ("Reply with only a single word: yes or no. Is Paris the capital of France?",
     lambda r: r.strip().lower() in ("yes", "no"),
     "single word yes/no"),
    ("List the planets of the solar system. Use a numbered list.",
     lambda r: "1." in r or "1)" in r,
     "numbered list"),
    ("Translate 'hello world' to Spanish. Reply in lowercase only.",
     lambda r: r == r.lower(),
     "lowercase only"),
    ("Write a haiku about the ocean. Format: line1 / line2 / line3",
     lambda r: "/" in r,
     "haiku with /"),
]

print(f"Loading model: {MODEL_PATH} (CPU, may take ~2-3 min)...", flush=True)
from gptqmodel import GPTQModel
from transformers import AutoTokenizer
import torch

t0 = time.time()
model = GPTQModel.load(MODEL_PATH, device="cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

passed = 0
for prompt, check_fn, label in PROMPTS:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    # Append /no_think hint if the template supports it
    if "<think>" not in text:
        text = text  # already no-think
    inputs = tokenizer(text, return_tensors="pt")
    t1 = time.time()
    with torch.no_grad():
        out_ids = model.model.generate(
            **inputs, max_new_tokens=512, temperature=None,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t1
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    out_clean = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL).strip()
    ok = check_fn(out_clean)
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"\n[{status}] {label} ({elapsed:.1f}s)")
    print(f"  Q: {prompt[:80]}...")
    print(f"  A: {out_clean[:200]}")

print(f"\n=== {passed}/{len(PROMPTS)} instruction-following tests passed ===")
if passed >= 4:
    print("Quality looks good for IFEval (≥4/5)")
elif passed >= 3:
    print("Marginal quality — may fail IFEval threshold (0.814)")
else:
    print("Low quality — likely to fail IFEval threshold (0.814)")
