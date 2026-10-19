"""Factual rubric and inference bookkeeping regression tests, without model weights."""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from lib.dataset import DEFAULT_RUST_SYSTEM_PROMPT, format_chatml_prompts
from scripts import eval_version_knowledge as evaluation


@pytest.fixture
def benchmark():
    return evaluation.load_benchmark()


def test_references_pass(benchmark):
    assert len(benchmark) == 22
    assert all(evaluation.score_item(i["reference_answer"], i)["rubric_pass"] for i in benchmark)


@pytest.mark.parametrize(
    "response,passed",
    [
        ("Rust 1.80.", True),
        ("1.80.0", True),
        ("LazyLock", False),
        ("Rust 1.75.0", False),
        ("Not 1.80.0", False),
        ("It was not stabilized in Rust 1.80.0.", False),
        ("Maybe Rust 1.80.", False),
        ("Rust 1.80.0 or 1.75.0", False),
        ("[LazyLock](https://example.org/1.80.0)", False),
        ("It wasn't 1.80.0.", False),
    ],
)
def test_reverse_lookup(response, passed, benchmark):
    item = next(i for i in benchmark if i["id"] == "reverse_lazylock")
    assert evaluation.score_item(response, item)["rubric_pass"] is passed


@pytest.mark.parametrize(
    "response,passed",
    [
        ("No. LazyLock requires Rust 1.80.", True),
        ("Yes. LazyLock requires Rust 1.80.", False),
        ("I do not know, 1.80.", False),
        ("No. It was stable in 1.75.", False),
        ("No. It was not stabilized in Rust 1.80.", False),
        ("No. Requires 1.80. Yes, it works.", False),
        ("No. It stabilized in Rust 1.75, rather than 1.80.", False),
    ],
)
def test_msrv(response, passed, benchmark):
    item = next(i for i in benchmark if i["id"] == "msrv_lazylock")
    assert evaluation.score_item(response, item)["rubric_pass"] is passed


def test_bounded_aliases_and_question_echo(benchmark):
    assert not evaluation.contains("spinning", "Pin")
    assert not evaluation.contains("i do not know", "No")
    item = next(i for i in benchmark if i["id"] == "msrv_core_error")
    assert not evaluation.score_item(item["question"], item)["rubric_pass"]
    assert evaluation.score_item("Yes.", item)["rubric_pass"]
    assert not evaluation.score_item("Yes. It stabilized in Rust 1.99.", item)["rubric_pass"]
    async_item = next(i for i in benchmark if i["id"] == "diff_1.74_1.75")
    response = "Stabilize `async fn` and return-position `impl Trait` in traits."
    assert evaluation.score_item(response, async_item)["rubric_pass"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: [],
        lambda rows: rows + [rows[0]],
        lambda rows: [{**rows[0], "facts": []}],
        lambda rows: [{**rows[0], "facts": [{"id": "x", "any_of": [""]}]}],
        lambda rows: [{**rows[0], "sources": []}],
        lambda rows: [{**rows[0], "reference_answer": "incorrect"}],
        lambda rows: [{**rows[0], "benchmark_version": 1}],
    ],
)
def test_invalid_benchmarks(tmp_path, benchmark, mutation):
    path = tmp_path / "benchmark.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in mutation(benchmark)))
    with pytest.raises(ValueError):
        evaluation.load_benchmark(path)


def test_malformed_json(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text("{")
    with pytest.raises(ValueError):
        evaluation.load_benchmark(path)


def test_rescore_matches_by_id_and_requires_regeneration(benchmark):
    saved = {
        "results": [
            {"id": i["id"], "question": i["question"], "response": i["reference_answer"]}
            for i in reversed(benchmark)
        ]
    }
    saved["results"][0]["question"] = "an old question"
    results = evaluation.rescore(saved, benchmark)
    assert results[0]["id"] == benchmark[0]["id"]
    assert results[-1]["status"] == "requires_regeneration"
    assert evaluation.summarize(results)["requires_regeneration"] == 1
    assert evaluation.summarize(results)["rubric_pass_rate"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.rescore({"results": saved["results"] + [saved["results"][0]]}, benchmark)
    with pytest.raises(ValueError, match="ID sets"):
        evaluation.rescore({"results": saved["results"][:-1]}, benchmark)
    with pytest.raises(ValueError):
        evaluation.rescore({"results": []}, benchmark)


def test_prompt_matches_training():
    prompt = evaluation.build_prompt("Question")
    training = format_chatml_prompts(
        {
            "instruction": ["Question"],
            "input": [""],
            "output": ["ANSWER"],
        }
    )["text"][0]
    assert DEFAULT_RUST_SYSTEM_PROMPT in prompt
    assert prompt + "ANSWER<|im_end|>" == training
    assert prompt.endswith("<|im_start|>assistant\n")


def test_completion_slicing_and_eos():
    tokenizer = MagicMock(eos_token_id=9)
    tokenizer.decode.return_value = " answer "
    tokens = MagicMock()
    tokens.__getitem__.return_value.tolist.return_value = [7, 9, 9]
    result = evaluation.decode_completion(tokenizer, tokens, 15, 3)
    tokens.__getitem__.assert_called_once_with(slice(15, None))
    tokenizer.decode.assert_called_once_with([7, 9], skip_special_tokens=True)
    assert result == {"response": "answer", "generated_tokens": 2, "finish_reason": "eos"}
    tokens.__getitem__.return_value.tolist.return_value = [1, 2, 3]
    assert evaluation.decode_completion(tokenizer, tokens, 15, 3)["finish_reason"] == "length"


def test_output_never_overwritten(tmp_path):
    path = tmp_path / "report.json"
    evaluation.write_report(path, {"original": True})
    with pytest.raises(FileExistsError):
        evaluation.write_report(path, {})
    assert json.loads(path.read_text()) == {"original": True}


def test_comparison_rejects_mismatched_runs(benchmark):
    item = benchmark[0]
    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        **evaluation.score_item(item["reference_answer"], item),
    }
    base = {
        "benchmark_sha256": "hash",
        "benchmark_ids": [item["id"]],
        "rubric_version": 2,
        "model_path": "model",
        "generation": {"max_new_tokens": 256},
        "load_in_4bit": False,
        "packages": {},
        "adapter_path": None,
        "results": [row],
    }
    adapter = copy.deepcopy(base)
    adapter["adapter_path"] = "checkpoint"
    assert evaluation.compare_reports(base, adapter)["paired_changes"][0]["rubric_pass_delta"] == 0
    adapter["generation"]["max_new_tokens"] = 512
    with pytest.raises(ValueError, match="generation"):
        evaluation.compare_reports(base, adapter)
    adapter = copy.deepcopy(base)
    adapter["adapter_path"] = "checkpoint"
    base["benchmark_ids"].append("missing")
    adapter["benchmark_ids"].append("missing")
    with pytest.raises(ValueError, match="complete"):
        evaluation.compare_reports(base, adapter)


def test_negated_feature_and_wrong_release_fail(benchmark):
    item = next(i for i in benchmark if i["id"] == "diff_1.64_1.65")
    for response in (
        "Rust 1.65 does not support let-else or Generic Associated Types.",
        "Rust 1.80 stabilized let-else and GATs.",
    ):
        assert not evaluation.score_item(response, item)["rubric_pass"]


def test_rescore_retains_truncation(benchmark):
    saved = {
        "results": [
            {
                "id": i["id"],
                "question": i["question"],
                "response": i["reference_answer"],
                "finish_reason": "length",
                "generated_tokens": 256,
            }
            for i in benchmark
        ]
    }
    results = evaluation.rescore(saved, benchmark)
    assert all(not r["rubric_pass"] for r in results)
    assert all("output_token_limit" in r["review_reasons"] for r in results)
    assert results[0]["generated_tokens"] == 256


def test_dataset_benchmark_overlap_audit(tmp_path, benchmark):
    row = {
        "instruction": benchmark[0]["question"],
        "input": "",
        "output": benchmark[0]["reference_answer"],
    }
    for name in ("train", "val"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
    audit = evaluation.audit_dataset(tmp_path, benchmark)
    assert audit["train"]["exact_normalized_question_hits"] == {benchmark[0]["id"]: 1}
    assert audit["val"]["exact_normalized_reference_hits"] == {benchmark[0]["id"]: 1}


def test_mocked_batched_inference_uses_training_chatml(benchmark):
    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    tokenizer = MagicMock(pad_token_id=9, eos_token_id=9)
    tokenizer.return_value = Inputs(
        input_ids=torch.tensor([[1, 2, 3], [9, 1, 2]]),
        attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
    )
    tokenizer.decode.side_effect = ["Rust 1.70.", "Rust 1.80."]
    model = MagicMock(device="cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3, 7, 9], [9, 1, 2, 8, 9]])
    items = [i for i in benchmark if i["id"] in ("reverse_oncelock", "reverse_lazylock")]
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("torch.cuda.is_available", return_value=False),
    ):
        report = evaluation.evaluate_model("test-model", None, items, batch_size=2)
    assert all(r["rubric_pass"] for r in report["results"])
    assert tokenizer.padding_side == "left"
    prompts = tokenizer.call_args.args[0]
    assert prompts == [evaluation.build_prompt(i["question"]) for i in items]
    assert tokenizer.call_args.kwargs["add_special_tokens"] is False
    assert model.generate.call_args.kwargs["eos_token_id"] == 9
    assert model.generate.call_args.kwargs["do_sample"] is False
    tokenizer.decode.assert_any_call([7, 9], skip_special_tokens=True)
    tokenizer.decode.assert_any_call([8, 9], skip_special_tokens=True)
    assert report["generation"]["system_prompt"] == DEFAULT_RUST_SYSTEM_PROMPT
