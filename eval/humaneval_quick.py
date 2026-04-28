"""Quick coding-eval harness.

Hits the LiteLLM gateway, runs a small fixed set of HumanEval-style problems,
executes the generated function in a subprocess with a timeout, and writes a
pass@1 + latency summary.

Usage:
    python eval/humaneval_quick.py --model coder-lg --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# -- tiny problem set ---------------------------------------------------------
PROBLEMS = [
    {
        "name": "fizzbuzz",
        "prompt": "Write a Python function `fizzbuzz(n)` that returns a list of strings for 1..n where multiples of 3 are 'Fizz', 5 are 'Buzz', and 15 are 'FizzBuzz'. Otherwise the number as a string.",
        "tests": "assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'",
    },
    {
        "name": "is_prime",
        "prompt": "Write a Python function `is_prime(n: int) -> bool`.",
        "tests": "assert is_prime(2) and is_prime(13) and not is_prime(1) and not is_prime(15)",
    },
    {
        "name": "reverse_words",
        "prompt": "Write `reverse_words(s: str) -> str` that reverses the order of whitespace-separated words, collapsing repeated whitespace to single spaces.",
        "tests": "assert reverse_words('  hello   world ') == 'world hello'",
    },
    {
        "name": "two_sum",
        "prompt": "Write `two_sum(nums: list[int], target: int) -> tuple[int,int] | None` returning indices (i,j) with i<j whose values sum to target, or None.",
        "tests": "assert two_sum([2,7,11,15], 9) == (0,1)\nassert two_sum([1,2,3], 7) is None",
    },
    {
        "name": "flatten",
        "prompt": "Write `flatten(xs)` that recursively flattens an arbitrarily nested list of ints into a flat list.",
        "tests": "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]",
    },
    {
        "name": "balanced",
        "prompt": "Write `balanced(s: str) -> bool` that returns True iff the brackets () [] {} in s are balanced.",
        "tests": "assert balanced('([]{})') and not balanced('([)]')",
    },
    {
        "name": "anagram",
        "prompt": "Write `is_anagram(a: str, b: str) -> bool` that ignores case and non-letters.",
        "tests": "assert is_anagram('Listen!', 'Silent') and not is_anagram('a','ab')",
    },
    {
        "name": "fib",
        "prompt": "Write `fib(n: int) -> int` returning the nth Fibonacci number (fib(0)=0, fib(1)=1).",
        "tests": "assert fib(0)==0 and fib(1)==1 and fib(10)==55",
    },
    {
        "name": "rle",
        "prompt": "Write `rle(s: str) -> str` returning run-length encoding like 'aaabbc' -> 'a3b2c1'.",
        "tests": "assert rle('aaabbc') == 'a3b2c1'\nassert rle('') == ''",
    },
    {
        "name": "sort_by_freq",
        "prompt": "Write `sort_by_freq(xs: list[int]) -> list[int]` sorting by descending frequency, ties broken by first appearance.",
        "tests": "assert sort_by_freq([4,5,6,5,4,3]) == [4,4,5,5,6,3]",
    },
    {
        "name": "word_count",
        "prompt": "Write `word_count(s: str) -> dict[str,int]` counting whitespace-separated words case-insensitively.",
        "tests": "assert word_count('a A b') == {'a':2,'b':1}",
    },
    {"name":"gcd","prompt":"Write `gcd(a:int,b:int)->int` (non-negative).","tests":"assert gcd(12,18)==6 and gcd(0,5)==5"},
    {"name":"transpose","prompt":"Write `transpose(m: list[list[int]]) -> list[list[int]]`.","tests":"assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]"},
    {"name":"unique","prompt":"Write `unique(xs)` preserving order, removing duplicates.","tests":"assert unique([1,2,1,3,2,4]) == [1,2,3,4]"},
    {"name":"max_subarray","prompt":"Write `max_subarray(nums: list[int]) -> int` (Kadane).","tests":"assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6"},
    {"name":"roman","prompt":"Write `to_roman(n: int) -> str` for 1..3999.","tests":"assert to_roman(1994) == 'MCMXCIV' and to_roman(9) == 'IX'"},
    {"name":"caesar","prompt":"Write `caesar(s: str, k: int) -> str` shifting ASCII letters by k, preserving case and non-letters.","tests":"assert caesar('Abc-Z', 1) == 'Bcd-A'"},
    {"name":"merge","prompt":"Write `merge(a: list[int], b: list[int]) -> list[int]` merging two sorted lists.","tests":"assert merge([1,3,5],[2,4,6]) == [1,2,3,4,5,6]"},
    {"name":"power_set","prompt":"Write `power_set(xs: list[int]) -> list[list[int]]` returning all subsets in any order.","tests":"r=power_set([1,2]); assert sorted(map(tuple,r)) == [(),(1,),(1,2),(2,)]"},
    {"name":"is_palindrome","prompt":"Write `is_palindrome(s: str) -> bool` ignoring case and non-alphanumerics.","tests":"assert is_palindrome('A man, a plan, a canal: Panama')"},
]

CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


def run_one(client: OpenAI, model: str, prob: dict) -> dict:
    sys_msg = "You are an expert Python programmer. Reply with ONLY a single ```python code block defining the requested function. No prose, no examples."
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prob["prompt"]},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    dt = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    code = extract_code(content)
    src = code + "\n\n" + prob["tests"] + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=5,
        )
        ok = proc.returncode == 0
        err = proc.stderr.strip().splitlines()[-1] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    finally:
        os.unlink(path)

    usage = getattr(resp, "usage", None)
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    return {
        "name": prob["name"], "ok": ok, "err": err,
        "latency_s": round(dt, 3),
        "out_tok": out_tok,
        "tok_per_s": round(out_tok / dt, 1) if dt > 0 and out_tok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=len(PROBLEMS))
    ap.add_argument("--base-url", default=os.getenv("LITELLM_URL", "http://192.168.3.5:4000/v1"))
    ap.add_argument("--api-key", default=os.getenv("LITELLM_KEY", "sk-local"))
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    out_dir = Path("eval/runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    results = []
    with results_path.open("w") as fh:
        for prob in PROBLEMS[: args.n]:
            r = run_one(client, args.model, prob)
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            mark = "OK " if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['name']:<16} {r['latency_s']}s  {r['tok_per_s']} tok/s  {r['err']}")

    n = len(results)
    passed = sum(1 for r in results if r["ok"])
    lats = sorted(r["latency_s"] for r in results)
    summary = {
        "model": args.model,
        "n": n,
        "pass@1": round(passed / n, 3),
        "p50_latency_s": lats[n // 2],
        "p95_latency_s": lats[max(0, int(n * 0.95) - 1)],
        "mean_tok_per_s": round(
            sum(r["tok_per_s"] for r in results if r["tok_per_s"]) /
            max(1, sum(1 for r in results if r["tok_per_s"])), 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n", json.dumps(summary, indent=2))
    print("→", out_dir)


if __name__ == "__main__":
    main()
