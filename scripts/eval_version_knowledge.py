#!/usr/bin/env python3
"""
Evaluate model and LoRA adapter knowledge of Rust editions, minor releases, diffs, and MSRV.
Runs against data/datasets/version_eval_benchmark.jsonl.
Evaluates accuracy based on key milestone recognition, semver resolution, and keyword hits.

Usage:
    # Run benchmark inspection / preflight validation:
    uv run python scripts/eval_version_knowledge.py --benchmark-only

    # Run evaluation against a local model or LoRA checkpoint:
    uv run python scripts/eval_version_knowledge.py --model-path Qwen/Qwen3.5-4B --adapter-path models/qwen3_5_4b_rust_lora
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.version_eval")

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_FILE = BASE_DIR / "data" / "datasets" / "version_eval_benchmark.jsonl"


def load_benchmark() -> list[dict[str, Any]]:
    if not BENCHMARK_FILE.exists():
        logger.error(f"Benchmark file not found: {BENCHMARK_FILE}")
        return []
    items: list[dict[str, Any]] = []
    with open(BENCHMARK_FILE) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def score_response(response: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """Score model response against expected keywords."""
    resp_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in resp_lower]
    score = len(matched) / len(expected_keywords) if expected_keywords else 1.0
    return score, matched


def run_benchmark_summary(items: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(" RUST VERSION & EDITION KNOWLEDGE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Total benchmark questions: {len(items)}\n")

    types: dict[str, int] = {}
    for item in items:
        q_type = item.get("type", "unknown")
        types[q_type] = types.get(q_type, 0) + 1

    print("Question Distribution:")
    for q_type, count in types.items():
        print(f"  - {q_type}: {count} questions")

    print("\nSample Benchmark Questions:")
    for idx, item in enumerate(items[:5], 1):
        print(f"\n[{idx}] ({item.get('type')}) {item.get('question')}")
        print(f"    Expected Keywords: {item.get('expected_keywords')}")
        print(f"    Reference: {item.get('reference_answer')[:120]}...")
    print("\n" + "=" * 60)


def evaluate_model(
    model_path: str,
    adapter_path: str | None,
    items: list[dict[str, Any]],
    device: str = "auto",
) -> None:
    """Evaluate model inference on the benchmark items."""
    logger.info(f"Loading model for evaluation: {model_path} (adapter: {adapter_path})")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers or torch not available for inference evaluation.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        logger.info(f"Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    total_score = 0.0
    results: list[dict[str, Any]] = []

    print("\nRunning benchmark evaluation...")
    for idx, item in enumerate(items, 1):
        question = item["question"]
        expected = item.get("expected_keywords", [])

        prompt = (
            f"<|im_start|>system\nYou are an expert Rust programmer with exact knowledge of Rust editions, rustc releases, and API stability.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )

        resp = tokenizer.decode(
            output_tokens[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        score, matched = score_response(resp, expected)
        total_score += score

        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "response": resp,
                "score": score,
                "matched_keywords": matched,
                "expected_keywords": expected,
            }
        )
        print(f"[{idx}/{len(items)}] Score: {score:.2f} | Question: {question[:50]}...")

    avg_score = total_score / len(items) if items else 0.0
    print("\n" + "=" * 60)
    print(f" EVALUATION RESULT: Overall Accuracy / Recall: {avg_score * 100:.1f}%")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Rust version knowledge on benchmark suite."
    )
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Print benchmark summary and exit."
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Base model path or HuggingFace ID."
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None, help="Path to LoRA adapter weights."
    )
    args = parser.parse_args()

    items = load_benchmark()
    if not items:
        return

    if args.benchmark_only or not args.model_path:
        run_benchmark_summary(items)
    else:
        evaluate_model(args.model_path, args.adapter_path, items)


if __name__ == "__main__":
    main()
