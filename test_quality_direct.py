#!/usr/bin/env python3
"""
Direct quality spot-check for the GPU submission container.
Tests representative questions from mmlu_pro, ifeval, and gpqa_diamond styles
without depending on lm_eval or HF datasets.

Usage:
  python3 test_quality_direct.py
"""

import json, re, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

CONTAINER_URL = "http://localhost:8080"

def ping():
    try:
        code = urllib.request.urlopen(f"{CONTAINER_URL}/ping", timeout=5).getcode()
        return code == 200
    except:
        return False

def invoke(prompt, max_tokens=256, temperature=0.0):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"{CONTAINER_URL}/invocations", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    text = r["choices"][0]["text"]
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def chat(messages, max_tokens=512, thinking=False):
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if thinking:
        payload["thinking"] = True
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{CONTAINER_URL}/invocations", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    text = r["choices"][0]["text"]
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── MMLU-Pro style (5-shot, multiple choice) ─────────────────────────────────
MMLU_QUESTIONS = [
    {
        "q": "What is the chemical formula for water?",
        "choices": ["A. H2O2", "B. H2O", "C. HO", "D. H3O"],
        "answer": "B",
    },
    {
        "q": "Which of the following is NOT a prime number?",
        "choices": ["A. 7", "B. 11", "C. 15", "D. 17"],
        "answer": "C",
    },
    {
        "q": "The derivative of sin(x) is:",
        "choices": ["A. -cos(x)", "B. cos(x)", "C. -sin(x)", "D. tan(x)"],
        "answer": "B",
    },
    {
        "q": "Which planet has the most moons?",
        "choices": ["A. Jupiter", "B. Saturn", "C. Uranus", "D. Neptune"],
        "answer": "B",
    },
    {
        "q": "In Python, what does `len([1,2,3])` return?",
        "choices": ["A. 2", "B. 4", "C. 3", "D. 1"],
        "answer": "C",
    },
    {
        "q": "The speed of light in vacuum is approximately:",
        "choices": ["A. 3×10^6 m/s", "B. 3×10^8 m/s", "C. 3×10^10 m/s", "D. 3×10^4 m/s"],
        "answer": "B",
    },
    {
        "q": "What is the integral of x^2 dx?",
        "choices": ["A. x^3", "B. x^3/3 + C", "C. 2x + C", "D. x^2/2 + C"],
        "answer": "B",
    },
    {
        "q": "Which data structure uses LIFO order?",
        "choices": ["A. Queue", "B. Heap", "C. Stack", "D. Tree"],
        "answer": "C",
    },
    {
        "q": "DNA is composed of nucleotides containing which bases?",
        "choices": ["A. A, U, G, C", "B. A, T, G, C", "C. A, T, G, U", "D. A, B, G, C"],
        "answer": "B",
    },
    {
        "q": "Newton's second law states that F equals:",
        "choices": ["A. mv", "B. ma", "C. m/a", "D. v/t"],
        "answer": "B",
    },
]

def make_mmlu_prompt(item):
    choices_str = "\n".join(item["choices"])
    return (f"Question: {item['q']}\n{choices_str}\n"
            f"Answer the question by choosing the letter of the correct answer.\n"
            f"Answer:")

# ── IFEval style (instruction following) ──────────────────────────────────────
IFEVAL_TESTS = [
    {
        "prompt": "List exactly 3 programming languages. Format as a numbered list.",
        "check": lambda r: sum(1 for line in r.split("\n") if re.match(r"^\s*[123]\.", line)) >= 3,
        "desc": "numbered list of 3 items",
    },
    {
        "prompt": "Write a one-sentence definition of machine learning. End your response with exactly '###'.",
        "check": lambda r: r.rstrip().endswith("###"),
        "desc": "ends with ###",
    },
    {
        "prompt": "Respond with only the word 'YES' in all caps and nothing else.",
        "check": lambda r: r.strip() == "YES",
        "desc": "exactly 'YES'",
    },
    {
        "prompt": "Write a haiku (5-7-5 syllables) about the sun. Put each line on its own line.",
        "check": lambda r: len([l for l in r.strip().split("\n") if l.strip()]) >= 3,
        "desc": "at least 3 lines",
    },
    {
        "prompt": "Give me a JSON object with keys 'name' and 'age'. Use valid JSON.",
        "check": lambda r: (lambda s: (lambda j: "name" in j and "age" in j)(json.loads(s)))(
            re.search(r"\{[^}]+\}", r).group() if re.search(r"\{[^}]+\}", r) else "{}"),
        "desc": "valid JSON with name and age",
    },
]

# ── GPQA-Diamond style (graduate-level reasoning) ─────────────────────────────
GPQA_TESTS = [
    {
        "q": ("A quantum harmonic oscillator has energy levels E_n = ℏω(n + 1/2). "
              "What is the zero-point energy (energy of the ground state, n=0)?"),
        "expected_pattern": r"ℏω/2|ℏω \* 1/2|half|½|0\.5.*ℏω|hbar.*omega.*half",
        "desc": "zero-point energy = ℏω/2",
    },
    {
        "q": ("In CRISPR-Cas9 gene editing, what molecule guides the Cas9 protein to the "
              "correct location in the genome?"),
        "expected_pattern": r"guide RNA|gRNA|sgRNA|single guide",
        "desc": "guide RNA / gRNA",
    },
    {
        "q": ("What is the Schwarzschild radius of a black hole with mass equal to Earth's mass "
              "(~6×10^24 kg)? Give the order of magnitude in mm or cm."),
        "expected_pattern": r"[0-9].*mm|millimeter|centimeter|cm|~9 mm|~0\.9 cm",
        "desc": "~9mm (order of magnitude)",
    },
]


def run_mmlu():
    print("\n── MMLU-Pro style (10 questions) ────────────────────────────────")
    correct = 0
    for i, item in enumerate(MMLU_QUESTIONS):
        prompt = make_mmlu_prompt(item)
        resp = invoke(prompt, max_tokens=10)
        # Extract answer letter
        match = re.search(r"\b([A-D])\b", resp[:20])
        predicted = match.group(1) if match else resp[:1].upper()
        ok = predicted == item["answer"]
        correct += ok
        mark = "✅" if ok else "❌"
        print(f"  [{i+1:2d}] {mark}  predicted={predicted!r} expected={item['answer']}  | {item['q'][:50]}")
    acc = correct / len(MMLU_QUESTIONS)
    print(f"  Accuracy: {correct}/{len(MMLU_QUESTIONS)} = {acc:.0%}")
    return acc


def run_ifeval():
    print("\n── IFEval style (5 instruction-following checks) ────────────────")
    passed = 0
    for i, test in enumerate(IFEVAL_TESTS):
        resp = chat([{"role": "user", "content": test["prompt"]}], max_tokens=200)
        try:
            ok = test["check"](resp)
        except Exception:
            ok = False
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"  [{i+1}] {mark}  check='{test['desc']}'")
        if not ok:
            print(f"       response: {resp[:80]!r}")
    rate = passed / len(IFEVAL_TESTS)
    print(f"  Pass rate: {passed}/{len(IFEVAL_TESTS)} = {rate:.0%}")
    return rate


def run_gpqa():
    print("\n── GPQA-Diamond style (3 graduate-level questions) ──────────────")
    passed = 0
    for i, test in enumerate(GPQA_TESTS):
        resp = chat([{"role": "user", "content": test["q"]}], max_tokens=300, thinking=True)
        ok = bool(re.search(test["expected_pattern"], resp, re.IGNORECASE))
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"  [{i+1}] {mark}  expect='{test['desc']}'")
        if not ok:
            print(f"       response: {resp[:120]!r}")
    rate = passed / len(GPQA_TESTS)
    print(f"  Pass rate: {passed}/{len(GPQA_TESTS)} = {rate:.0%}")
    return rate


if __name__ == "__main__":
    print(f"Checking {CONTAINER_URL}/ping ...")
    if not ping():
        print("❌ Container not ready"); sys.exit(1)
    print("✅ Container ready\n")

    t0 = time.perf_counter()
    mmlu_acc  = run_mmlu()
    ifeval_rate = run_ifeval()
    gpqa_rate   = run_gpqa()
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"QUALITY SUMMARY  ({elapsed:.1f}s total)")
    print(f"{'='*60}")
    print(f"  MMLU-Pro style  : {mmlu_acc:.0%}  (threshold ~62%)")
    print(f"  IFEval style    : {ifeval_rate:.0%}  (threshold ~81%)")
    print(f"  GPQA style      : {gpqa_rate:.0%}  (threshold ~63%)")

    mmlu_pass   = mmlu_acc   >= 0.60
    ifeval_pass = ifeval_rate >= 0.60   # easier threshold for spot-check
    gpqa_pass   = gpqa_rate  >= 0.60

    all_pass = mmlu_pass and ifeval_pass and gpqa_pass
    print(f"\n  Overall: {'✅ PASS' if all_pass else '⚠️  PARTIAL'}")
    sys.exit(0 if all_pass else 1)
