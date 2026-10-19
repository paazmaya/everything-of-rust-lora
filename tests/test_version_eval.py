"""Factual rubric and inference bookkeeping regression tests, without model weights."""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from lib import version_eval as evaluation
from lib.dataset import DEFAULT_RUST_SYSTEM_PROMPT, format_chatml_prompts


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


def test_cli_modes_and_output_safety(tmp_path, benchmark, monkeypatch, capsys):
    def run(*args):
        monkeypatch.setattr("sys.argv", ["eval_version_knowledge.py", *map(str, args)])
        evaluation.main()

    run("--benchmark-only")
    assert json.loads(capsys.readouterr().out)["questions"] == 22
    run()
    assert json.loads(capsys.readouterr().out)["reference_facts_validated"] is True
    base_path = tmp_path / "base.json"
    payload = {
        "model_path": "test-model",
        "adapter_path": None,
        "generation": {},
        "packages": {},
        "load_in_4bit": False,
        "results": [
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "response": item["reference_answer"],
                **evaluation.score_item(item["reference_answer"], item),
            }
            for item in benchmark
        ],
    }
    with patch.object(evaluation, "evaluate_model", return_value=payload) as inference:
        run("--model-path", "test-model", "--output-file", base_path)
        inference.assert_called_once()
    base = json.loads(base_path.read_text())
    assert base["summary"]["rubric_pass_count"] == 22
    assert len(base["benchmark_ids"]) == 22
    with pytest.raises(FileExistsError):
        run("--model-path", "test-model", "--output-file", base_path)
    adapter_path = tmp_path / "adapter.json"
    evaluation.write_report(adapter_path, {**base, "adapter_path": "checkpoint"})
    comparison_path = tmp_path / "comparison.json"
    run("--compare-files", base_path, adapter_path, "--output-file", comparison_path)
    assert len(json.loads(comparison_path.read_text())["paired_changes"]) == 22
    rescored = tmp_path / "rescored.json"
    run("--rescore-file", base_path, "--output-file", rescored)
    assert json.loads(rescored.read_text())["summary"]["rubric_pass_count"] == 22
    for name in ("train", "val"):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(
                {
                    "instruction": "unrelated",
                    "input": "",
                    "output": "unrelated",
                }
            )
            + "\n"
        )
    audit_path = tmp_path / "audit.json"
    run("--audit-dataset-dir", tmp_path, "--output-file", audit_path)
    assert json.loads(audit_path.read_text())["train"]["rows"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--batch-size", "0"],
        ["--max-new-tokens", "0"],
        ["--rescore-file", "old.json", "--model-path", "model"],
        ["--rescore-file", "old.json", "--benchmark-only"],
        ["--adapter-path", "checkpoint"],
    ],
)
def test_cli_rejects_invalid_options(args, monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval_version_knowledge.py", *args])
    with pytest.raises(SystemExit):
        evaluation.main()


@pytest.mark.parametrize(
    "args",
    [
        ["--compare-files", "a.json", "b.json", "--model-path", "model"],
        ["--audit-dataset-dir", "data", "--model-path", "model"],
    ],
)
def test_cli_rejects_conflicting_modes(args, monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval_version_knowledge.py", *args])
    with pytest.raises(ValueError):
        evaluation.main()


def test_legacy_diagnostic_and_bad_saved_rows(benchmark):
    assert evaluation.score_response("OnceLock", ["OnceLock", "1.70.0"]) == (0.5, ["OnceLock"])
    assert evaluation.score_response("text", []) == (0.0, [])
    for saved in (
        {"results": [None]},
        {"results": [{"id": "item"}]},
    ):
        with pytest.raises(ValueError):
            evaluation.rescore(saved, benchmark)


@pytest.mark.parametrize(
    "replacement",
    [
        {"id": ""},
        {"question_revision": 0},
        {"facts": [{"id": "", "any_of": ["Rust"]}]},
        {"facts": [{"id": "x", "any_of": ["Rust"]}, {"id": "x", "any_of": ["Rust"]}]},
        {"type": "reverse_lookup", "answer_version": "1.80"},
        {"type": "msrv_check", "decision": "perhaps"},
        {"forbidden": "text"},
        {"context_versions": ["nonsense"]},
        {"required_version": "1.80"},
    ],
)
def test_more_invalid_rubric_fields(tmp_path, benchmark, replacement):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({**benchmark[0], **replacement}))
    with pytest.raises(ValueError):
        evaluation.load_benchmark(path)


def test_benchmark_nonobject_and_empty_lines(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("\nnull\n")
    with pytest.raises(ValueError, match="expected an object"):
        evaluation.load_benchmark(path)


def test_offline_legacy_scores_and_question_mismatches(benchmark):
    old = {
        "results": [
            {
                "id": i["id"],
                "question": i["question"],
                "response": i["reference_answer"],
                "expected_keywords": ["Rust"],
            }
            for i in benchmark
        ]
    }
    results = evaluation.rescore(old, benchmark)
    assert all("legacy_keyword_recall" in r for r in results)
    results = [
        {
            "id": "x",
            "question": "one",
            "type": "reverse_lookup",
            "rubric_pass": False,
            "fact_coverage": 0.0,
            "review_reasons": [],
        },
    ]
    base = {
        "benchmark_sha256": "hash",
        "benchmark_ids": ["x"],
        "rubric_version": 2,
        "model_path": "model",
        "generation": {},
        "packages": {},
        "load_in_4bit": False,
        "adapter_path": None,
        "results": results,
    }
    adapter = copy.deepcopy(base)
    adapter["adapter_path"] = "checkpoint"
    adapter["results"][0]["question"] = "two"
    with pytest.raises(ValueError, match="Question mismatch"):
        evaluation.compare_reports(base, adapter)
    adapter = copy.deepcopy(base)
    with pytest.raises(ValueError, match="base report"):
        evaluation.compare_reports(base, adapter)


def test_gpu_adapter_loading_and_truncation(benchmark):
    tokenizer = MagicMock(eos_token_id=9, pad_token_id=None)
    tokenizer.decode.return_value = "Rust 1.70.0"
    model = MagicMock(device="cpu")
    inputs = MagicMock()
    inputs.__getitem__.return_value.shape = [1, 3]
    tokenizer.return_value.to.return_value = inputs
    tokens = MagicMock()
    tokens.__getitem__.return_value.__getitem__.return_value.tolist.return_value = [7] * 2
    model.generate.return_value = tokens
    item = next(i for i in benchmark if i["id"] == "reverse_oncelock")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("transformers.BitsAndBytesConfig") as quantization,
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load,
        patch("peft.PeftModel.from_pretrained", return_value=model) as adapter,
    ):
        report = evaluation.evaluate_model("model", "checkpoint", [item], max_new_tokens=2)
    assert report["load_in_4bit"] is True
    assert load.call_args.kwargs["device_map"] == {"": 0}
    quantization.assert_called_once()
    adapter.assert_called_once_with(model, "checkpoint")
    assert report["results"][0]["rubric_pass"] is False
    assert "output_token_limit" in report["results"][0]["review_reasons"]
    tokenizer.eos_token_id = None
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        with pytest.raises(ValueError, match="EOS"):
            evaluation.evaluate_model("model", None, [item])
    with pytest.raises(ValueError, match="Nonempty"):
        evaluation.evaluate_model("model", None, [])
