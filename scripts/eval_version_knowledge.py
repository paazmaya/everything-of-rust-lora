#!/usr/bin/env python3
"""Source-backed Rust factual evaluation and offline rescoring (no LLM judge).

Automated rubric passes are not a guarantee that every statement is true.
Unknown paraphrases and extra claims need manual review; coverage is not accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from lib.config import DEFAULT_BENCHMARK_FILE
from lib.dataset import CHATML_PROMPT_TEMPLATE, DEFAULT_RUST_SYSTEM_PROMPT

logger = logging.getLogger("rust_lora.version_eval")
BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = BASE_DIR / DEFAULT_BENCHMARK_FILE
RUBRIC_VERSION = 2
CATEGORIES = {"consecutive_diff", "reverse_lookup", "edition_guide", "msrv_check"}
VERSION_RE = re.compile(r"(?<![\w.])(1\.\d+)(?:\.(\d+))?(?!\w|\.\d)")


def normalize(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.lower().replace("`", "").replace("**", "")
    text = text.replace("\u2011", "-").replace("\u2013", "-")
    text = VERSION_RE.sub(lambda m: f"{m[1]}.{m[2] or '0'}", text)
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, alias: str) -> bool:
    pattern = re.escape(normalize(alias)).replace(r"\ ", r"\s+")
    return re.search(r"(?<!\w)" + pattern + r"(?!\w)", text) is not None


def score_response(response: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """Historical substring recall ONLY, retained for interpreting old reports."""
    matched = [kw for kw in expected_keywords if kw.lower() in response.lower()]
    return len(matched) / len(expected_keywords) if expected_keywords else 0.0, matched


def version_checks(text: str, item: dict[str, Any]) -> tuple[list[str], list[str]]:
    contradictions, review = [], []
    expected_version = item.get("answer_version")
    versions = VERSION_RE.findall(text)
    mentioned_versions = {f"{minor}.{patch or '0'}" for minor, patch in versions}
    if item.get("allowed_versions") and mentioned_versions - set(item["allowed_versions"]):
        contradictions.append("unexpected_release_version")
    if expected_version:
        allowed = {expected_version, *item.get("context_versions", [])}
        if mentioned_versions - allowed:
            contradictions.append("unexpected_answer_version")
        # Additional context versions are legitimate, but their relationship needs review.
        if len(mentioned_versions) > 1:
            review.append("multiple_versions_require_relationship_review")
    if item.get("required_version") and not contains(text, item["required_version"]):
        contradictions.append("missing_required_stabilization_version")
    expected_stabilization = item.get("answer_version") or item.get("required_version")
    if expected_stabilization:
        assertions = re.findall(
            r"\b(?:stabilized|introduced)\s+(?:in\s+)?(?:rust\s+)?(1\.\d+\.\d+)\b",
            text,
        )
        if any(version != expected_stabilization for version in assertions):
            review.append("conflicting_stabilization_claim")
    return contradictions, review


def decision_checks(text: str, decision: str | None) -> tuple[bool, list[str], list[str]]:
    contradictions, review = [], []
    decision_match = re.match(r"^(?:answer:\s*)?(yes|no)\b", text)
    decision_correct = decision_match is not None and decision_match[1] == decision
    if decision:
        if not decision_match:
            review.append("explicit_yes_no_required")
        elif not decision_correct:
            contradictions.append("wrong_compatibility_decision")
        if re.search(
            r"\b(?:yes|no)[,.:!]", text[len(decision_match[0]) :] if decision_match else text
        ):
            review.append("multiple_decisions")
    return decision_correct, contradictions, review


def review_checks(text: str, item: dict[str, Any]) -> list[str]:
    review = []
    if re.search(r"\b(?:not sure|do not know|don't know|maybe|perhaps|might be)\b", text):
        review.append("uncertain_answer")
    # Conservative flags: do not silently certify a negated required fact or a
    # stabilization claim described as nightly-only. Human review resolves context.
    for fact in item["facts"]:
        for alias in fact["any_of"]:
            escaped = re.escape(normalize(alias))
            if re.search(
                rf"\b(?:not|never(?!\s+type\b)|without)(?:\s+\w+){{0,5}}\s+{escaped}(?!\w)",
                text,
            ) or re.search(rf"{escaped}\s+(?:is|was)\s+(?:not|never)\b", text):
                review.append(f"negated_fact:{fact['id']}")
    if re.search(r"\b(?:unstable|nightly-only|only available in nightly|wasn't|isn't)\b", text):
        review.append("unstable_claim_requires_review")
    return review


def score_item(response: str, item: dict[str, Any]) -> dict[str, Any]:
    text = normalize(response.replace(item["question"], ""))
    matched = [
        fact["id"]
        for fact in item["facts"]
        if any(contains(text, alias) for alias in fact["any_of"])
    ]
    contradictions, review = version_checks(text, item)
    decision = item.get("decision")
    decision_correct, decision_errors, decision_review = decision_checks(text, decision)
    contradictions.extend(decision_errors)
    contradictions.extend(
        f"forbidden_claim:{alias}" for alias in item.get("forbidden", []) if contains(text, alias)
    )
    review.extend(decision_review)
    review.extend(review_checks(text, item))
    if not matched and item["facts"]:
        review.append("no_required_facts")
    required_count = len(item["facts"]) + bool(decision)
    coverage = (len(matched) + bool(decision and decision_correct)) / required_count
    passed = coverage == 1.0 and not contradictions and not review
    return {
        "rubric_pass": passed,
        "fact_coverage": coverage,
        "matched_facts": matched,
        "missing_facts": [f["id"] for f in item["facts"] if f["id"] not in matched],
        "contradictions": contradictions,
        "review_reasons": sorted(set(review)),
    }


def validate_facts(item: dict[str, Any]) -> None:
    facts = item.get("facts")
    if not isinstance(facts, list) or (
        not facts and not (item["type"] == "msrv_check" and item.get("decision"))
    ):
        raise ValueError(f"{item['id']}: nonempty facts or a compatibility decision required")
    fact_ids = []
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str) or not fact["id"]:
            raise ValueError(f"{item['id']}: invalid fact")
        aliases = fact.get("any_of")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
        ):
            raise ValueError(f"{item['id']}: nonempty aliases required")
        fact_ids.append(fact["id"])
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError(f"{item['id']}: duplicate fact IDs")


def validate_item(item: dict[str, Any]) -> None:
    if item.get("type") not in CATEGORIES or item.get("benchmark_version") != RUBRIC_VERSION:
        raise ValueError(f"{item['id']}: invalid category or benchmark version")
    if not isinstance(item.get("question_revision"), int) or item["question_revision"] < 1:
        raise ValueError(f"{item['id']}: invalid question revision")
    if (
        not isinstance(item.get("sources"), list)
        or not item["sources"]
        or not all(isinstance(url, str) and url.startswith("https://") for url in item["sources"])
    ):
        raise ValueError(f"{item['id']}: source URLs required")
    validate_facts(item)
    validate_optional_fields(item)
    if item["type"] == "reverse_lookup" and not re.fullmatch(
        r"1\.\d+\.\d+", item.get("answer_version", "")
    ):
        raise ValueError(f"{item['id']}: answer_version required")
    if item["type"] == "msrv_check" and (item.get("decision") not in ("yes", "no")):
        raise ValueError(f"{item['id']}: compatibility decision required")
    if not score_item(item["reference_answer"], item)["rubric_pass"]:
        raise ValueError(f"{item['id']}: reference fails its own rubric")


def validate_optional_fields(item: dict[str, Any]) -> None:
    for key in ("forbidden", "context_versions", "allowed_versions"):
        if key in item and (
            not isinstance(item[key], list)
            or not all(isinstance(value, str) and value.strip() for value in item[key])
        ):
            raise ValueError(f"{item['id']}: invalid {key}")
    for key in ("context_versions", "allowed_versions"):
        if any(not re.fullmatch(r"1\.\d+\.\d+", v) for v in item.get(key, [])):
            raise ValueError(f"{item['id']}: invalid version in {key}")
    if "required_version" in item and not re.fullmatch(r"1\.\d+\.\d+", item["required_version"]):
        raise ValueError(f"{item['id']}: invalid required_version")


def load_benchmark(path: Path = BENCHMARK_FILE) -> list[dict[str, Any]]:
    items = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        for field in ("id", "question", "reference_answer"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{path}:{line_no}: invalid {field}")
        if item["id"] in seen:
            raise ValueError(f"Duplicate benchmark ID: {item['id']}")
        seen.add(item["id"])
        validate_item(item)
        items.append(item)
    if not items:
        raise ValueError(f"Empty benchmark: {path}")
    return items


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in results if r.get("status") != "requires_regeneration"]

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "count": count,
            "rubric_pass_count": sum(r["rubric_pass"] for r in rows),
            "rubric_pass_rate": sum(r["rubric_pass"] for r in rows) / count if count else None,
            "average_fact_coverage": sum(r["fact_coverage"] for r in rows) / count
            if count
            else None,
            "review_count": sum(bool(r["review_reasons"]) for r in rows),
        }

    return {
        **metrics(eligible),
        "requires_regeneration": len(results) - len(eligible),
        "categories": {
            category: metrics([r for r in eligible if r["type"] == category])
            for category in sorted(CATEGORIES)
        },
        "limitations": "Rubric passes are not semantic truth guarantees; inspect extra claims and paraphrases manually.",
    }


def saved_rows_by_id(saved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = saved.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Saved report must contain nonempty results")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Saved result requires an ID")
        if row["id"] in by_id:
            raise ValueError(f"Duplicate saved ID: {row['id']}")
        if not isinstance(row.get("response"), str) or not isinstance(row.get("question"), str):
            raise ValueError(f"{row['id']}: question and response required")
        by_id[row["id"]] = row
    return by_id


def rescore(saved: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = saved_rows_by_id(saved)
    if set(by_id) != {item["id"] for item in items}:
        raise ValueError("Saved report and benchmark ID sets differ")
    results = []
    for item in items:
        old = by_id[item["id"]]
        result = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "question_revision": item["question_revision"],
            "response": old["response"],
            "reference_answer": item["reference_answer"],
            "original_question": old["question"],
        }
        if old["question"] != item["question"]:
            result["status"] = "requires_regeneration"
        else:
            result.update(score_item(old["response"], item))
            result["status"] = "rescored"
            if old.get("finish_reason") == "length":
                result["review_reasons"].append("output_token_limit")
                result["rubric_pass"] = False
        for field in ("generated_tokens", "finish_reason"):
            if field in old:
                result[field] = old[field]
        if isinstance(old.get("expected_keywords"), list):
            result["legacy_keyword_recall"], _ = score_response(
                old["response"], old["expected_keywords"]
            )
        results.append(result)
    return results


def audit_dataset(directory: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    questions = {normalize(item["question"]): item["id"] for item in items}
    references = {normalize(item["reference_answer"]): item["id"] for item in items}
    report: dict[str, Any] = {
        "scope": "In-domain factual benchmark, not unseen-source generalization. Exact-match checks do not rule out paraphrases.",
    }
    for name in ("train", "val"):
        question_hits: Counter[str] = Counter()
        reference_hits: Counter[str] = Counter()
        rows = 0
        with (directory / f"{name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows += 1
                question = questions.get(normalize(row["instruction"]))
                reference = references.get(normalize(row["output"]))
                if question:
                    question_hits[question] += 1
                if reference:
                    reference_hits[reference] += 1
        report[name] = {
            "rows": rows,
            "exact_normalized_question_hits": dict(question_hits),
            "exact_normalized_reference_hits": dict(reference_hits),
        }
    return report


def build_prompt(question: str) -> str:
    return CHATML_PROMPT_TEMPLATE.format(
        system_prompt=DEFAULT_RUST_SYSTEM_PROMPT,
        user_message=question,
        assistant_message="",
    ).removesuffix("<|im_end|>")


def decode_completion(
    tokenizer: Any,
    tokens: Any,
    prompt_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated = tokens[prompt_length:].tolist()
    eos_id = tokenizer.eos_token_id
    ended = eos_id in generated
    if ended:
        generated = generated[: generated.index(eos_id) + 1]
    return {
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_tokens": len(generated),
        "finish_reason": "eos"
        if ended
        else "length"
        if len(generated) >= max_new_tokens
        else "other",
    }


def evaluate_model(
    model_path: str,
    adapter_path: str | None,
    items: list[dict[str, Any]],
    device: str = "auto",
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    if not items or batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Nonempty items and positive batch/token limits required")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    quantized = use_cuda and load_in_4bit
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs.update(dtype=torch.bfloat16 if use_cuda else torch.float32, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    results = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        inputs = tokenizer(
            [build_prompt(item["question"]) for item in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for offset, item in enumerate(batch):
            completion = decode_completion(
                tokenizer, tokens[offset], inputs["input_ids"].shape[1], max_new_tokens
            )
            result = {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "question_revision": item["question_revision"],
                "reference_answer": item["reference_answer"],
                **completion,
                **score_item(completion["response"], item),
            }
            if completion["finish_reason"] == "length":
                result["review_reasons"].append("output_token_limit")
                result["rubric_pass"] = False
            results.append(result)
    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "load_in_4bit": quantized,
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": tokenizer.eos_token_id,
            "prompt_format": "training_chatml",
            "system_prompt": DEFAULT_RUST_SYSTEM_PROMPT,
            "device": str(model.device),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
        },
        "results": results,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Evaluation artifacts must never silently replace original evidence.
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def compare_reports(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    for field in (
        "benchmark_sha256",
        "benchmark_ids",
        "rubric_version",
        "model_path",
        "generation",
        "load_in_4bit",
        "packages",
    ):
        if field not in base or base[field] != adapter.get(field):
            raise ValueError(f"Cannot compare mismatched or missing {field}")
    if base.get("adapter_path") or not adapter.get("adapter_path"):
        raise ValueError("Comparison requires a base report followed by an adapter report")
    left = base.get("results", [])
    right = adapter.get("results", [])
    left_ids = [r["id"] for r in left]
    right_ids = [r["id"] for r in right]
    if (
        not left_ids
        or len(set(left_ids)) != len(left_ids)
        or set(left_ids) != set(right_ids)
        or len(right_ids) != len(left_ids)
        or set(left_ids) != set(base["benchmark_ids"])
        or len(left_ids) != len(base["benchmark_ids"])
    ):
        raise ValueError("Comparison requires identical complete, unique ID sets")
    paired = {r["id"]: r for r in right}
    changes = []
    for row in left:
        other = paired[row["id"]]
        if row["question"] != other["question"] or row["type"] != other["type"]:
            raise ValueError(f"Question mismatch: {row['id']}")
        changes.append(
            {
                "id": row["id"],
                "type": row["type"],
                "rubric_pass_delta": int(other["rubric_pass"]) - int(row["rubric_pass"]),
                "fact_coverage_delta": other["fact_coverage"] - row["fact_coverage"],
            }
        )
    return {"base": summarize(left), "adapter": summarize(right), "paired_changes": changes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-file", type=Path, default=BENCHMARK_FILE)
    parser.add_argument("--rescore-file", type=Path)
    parser.add_argument("--audit-dataset-dir", type=Path)
    parser.add_argument(
        "--compare-files", type=Path, nargs=2, metavar=("BASE_REPORT", "ADAPTER_REPORT")
    )
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-file", type=Path, default=Path("version_eval_results.json"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")
    if args.rescore_file and (args.model_path or args.benchmark_only):
        parser.error("--rescore-file cannot be combined with model inference or benchmark-only")
    if args.adapter_path and not args.model_path:
        parser.error("--adapter-path requires --model-path")
    return args


def main() -> None:
    args = parse_args()
    if args.compare_files:
        if (
            args.model_path
            or args.rescore_file
            or args.benchmark_only
            or args.adapter_path
            or args.audit_dataset_dir
        ):
            raise ValueError("--compare-files cannot be combined with other evaluation modes")
        base, adapter = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare_files]
        comparison = compare_reports(base, adapter)
        write_report(args.output_file, comparison)
        print(json.dumps(comparison, indent=2))
        return
    items = load_benchmark(args.benchmark_file)
    if args.audit_dataset_dir:
        if args.model_path or args.rescore_file or args.adapter_path:
            raise ValueError("--audit-dataset-dir cannot be combined with inference or rescoring")
        report = audit_dataset(args.audit_dataset_dir, items)
        report["benchmark_sha256"] = hashlib.sha256(args.benchmark_file.read_bytes()).hexdigest()
        write_report(args.output_file, report)
        print(json.dumps(report, indent=2))
        return
    if args.benchmark_only or not (args.model_path or args.rescore_file):
        print(
            json.dumps(
                {
                    "questions": len(items),
                    "categories": dict(Counter(i["type"] for i in items)),
                    "benchmark_version": RUBRIC_VERSION,
                    "reference_facts_validated": True,
                },
                indent=2,
            )
        )
        return
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output_file}")
    if args.rescore_file:
        saved = json.loads(args.rescore_file.read_text(encoding="utf-8"))
        payload = {
            "source_report": str(args.rescore_file),
            "source_sha256": hashlib.sha256(args.rescore_file.read_bytes()).hexdigest(),
            "model_path": saved.get("model_path"),
            "adapter_path": saved.get("adapter_path"),
            "historical_generation_metadata": "unknown unless recorded in source_report",
            "source_generation": saved.get("generation"),
            "source_packages": saved.get("packages"),
            "load_in_4bit": saved.get("load_in_4bit"),
            "results": rescore(saved, items),
        }
    else:
        payload = evaluate_model(
            args.model_path,
            args.adapter_path,
            items,
            load_in_4bit=not args.no_4bit,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    payload.update(
        rubric_version=RUBRIC_VERSION,
        benchmark_ids=[item["id"] for item in items],
        benchmark_sha256=hashlib.sha256(args.benchmark_file.read_bytes()).hexdigest(),
        summary=summarize(payload["results"]),
    )
    write_report(args.output_file, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved report: {args.output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
