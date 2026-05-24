#!/usr/bin/env python3
"""
Verify thinking-mode control per benchmark type.
One prompt per scenario. Uses max_new_tokens=32 — fast on CPU.

Scenarios:
  MMLU-Pro  → enable_thinking=False  (no stream)
  IFEval    → enable_thinking=False  (no stream)
  GPQA      → enable_thinking=True   (stream=True)
"""
import sys, time
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "qwen-weights-gptq"

SCENARIOS = [
    {
        "bench":          "MMLU-Pro",
        "enable_thinking": False,
        "prompt":         "The chemical formula for water is: (A) H2O  (B) CO2  (C) NaCl  (D) O2. Answer:",
        "pass_if":        lambda out, think: "h2o" in out.lower() or "(a)" in out.lower(),
        "label":          "answer contains H2O or (A)",
    },
    {
        "bench":          "IFEval",
        "enable_thinking": False,
        "prompt":         "Reply with only a single word: yes or no. Is Paris the capital of France?",
        "pass_if":        lambda out, think: out.strip().lower() in ("yes", "no"),
        "label":          "single word yes/no",
    },
    {
        "bench":          "GPQA",
        "enable_thinking": True,
        "prompt":         "In quantum mechanics, what is Heisenberg's uncertainty principle? Very brief.",
        "pass_if":        lambda out, think: len(think) > 0,
        "label":          "thinking tokens present",
    },
]

print(f"Loading tokenizer + model from {MODEL_PATH} (CPU)...", flush=True)
from gptqmodel import GPTQModel
from transformers import AutoTokenizer
import torch, re

t0 = time.time()
model     = GPTQModel.load(MODEL_PATH, device="cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded in {time.time()-t0:.0f}s\n", flush=True)

# ── Step 1: Template rendering check (instant) ────────────────────────────────
print("=== Template rendering check ===")
messages = [{"role": "user", "content": "test"}]
for enabled, label in [(False, "enable_thinking=False"), (True, "enable_thinking=True")]:
    # Use direct kwarg — this is how vLLM calls apply_chat_template after
    # unpacking chat_template_kwargs from the API request body.
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enabled,
    )
    suffix = rendered.split("<|im_start|>assistant\n")[-1]
    if enabled:
        ok = suffix.startswith("<think>\n") and not suffix.startswith("<think>\n\n</think>")
        expected = "opens <think> for model to fill"
    else:
        ok = "<think>\n\n</think>" in suffix
        expected = "empty <think></think> block"
    print(f"  [{('OK' if ok else 'FAIL')}] {label}: {expected}")
    print(f"        suffix preview: {repr(suffix[:60])}")
print()

# ── Step 2: Inference check (max_new_tokens=32, fast) ────────────────────────
print("=== Inference check (max_new_tokens=32) ===")
passed = 0
for s in SCENARIOS:
    messages = [{"role": "user", "content": s["prompt"]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=s["enable_thinking"],
    )
    inputs = tokenizer(text, return_tensors="pt")
    t1 = time.time()
    with torch.no_grad():
        out_ids = model.model.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            temperature=None, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t1
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Extract thinking and answer parts
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    answer = re.sub(r"<[^>]+>", "", answer).strip()  # strip remaining special tokens

    ok = s["pass_if"](answer, think)
    if ok:
        passed += 1
    status = "PASS" if ok else "FAIL"

    print(f"\n[{status}] {s['bench']} — {s['label']} ({elapsed:.1f}s)")
    print(f"  enable_thinking={s['enable_thinking']}")
    print(f"  raw output:  {repr(raw[:120])}")
    print(f"  think block: {repr(think[:80]) if think else '(none)'}")
    print(f"  answer:      {repr(answer[:80])}")

print(f"\n=== {passed}/{len(SCENARIOS)} scenarios passed ===")
