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

from lib.config import DEFAULT_BENCHMARK_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.version_eval")

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_FILE = BASE_DIR / DEFAULT_BENCHMARK_FILE


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
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    output_file: Path | None = None,
) -> None:
    """Evaluate model inference on the benchmark items."""
    logger.info(f"Loading model for evaluation: {model_path} (adapter: {adapter_path})")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        logger.error("transformers or torch not available for inference evaluation.")
        return

    use_cuda = torch.cuda.is_available()
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_cuda and load_in_4bit:
        # Matches QLoRA training and keeps the whole model on the GPU (no CPU offload)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = {"": 0}
        logger.info("Loading base model in 4-bit (NF4) on GPU 0")
    else:
        load_kwargs["dtype"] = torch.bfloat16 if use_cuda else torch.float32
        load_kwargs["device_map"] = device

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if adapter_path:
        from peft import PeftModel

        logger.info(f"Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    total_score = 0.0
    results: list[dict[str, Any]] = []

    print("\nRunning benchmark evaluation...")
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [
            "<|im_start|>system\nYou are an expert Rust programmer with exact knowledge of Rust editions, rustc releases, and API stability.<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            for item in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for offset, item in enumerate(batch):
            idx = start + offset + 1
            question = item["question"]
            expected = item.get("expected_keywords", [])
            resp = tokenizer.decode(
                output_tokens[offset][prompt_len:], skip_special_tokens=True
            ).strip()
            score, matched = score_response(resp, expected)
            total_score += score

            results.append(
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "question": question,
                    "response": resp,
                    "reference_answer": item.get("reference_answer"),
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

    if output_file:
        payload = {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "load_in_4bit": bool(use_cuda and load_in_4bit),
            "average_score": avg_score,
            "results": results,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved per-question responses to: {output_file}")


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
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the base model in bf16 instead of 4-bit (needs enough VRAM for the full model).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Questions generated per batch [default: 4]."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max generated tokens per answer."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("version_eval_results.json"),
        help="JSON file for per-question responses and scores [default: version_eval_results.json].",
    )
    args = parser.parse_args()

    items = load_benchmark()
    if not items:
        return

    if args.benchmark_only or not args.model_path:
        run_benchmark_summary(items)
    else:
        evaluate_model(
            args.model_path,
            args.adapter_path,
            items,
            load_in_4bit=not args.no_4bit,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
