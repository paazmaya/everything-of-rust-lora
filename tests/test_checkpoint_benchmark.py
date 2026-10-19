"""Checkpoint probes must not change the training trajectory or lose run history."""

import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import BatchEncoding, TrainerControl, TrainerState

from lib.checkpoint_benchmark import (
    MAX_NEW_TOKENS,
    PROBE_IDS,
    CheckpointBenchmarkCallback,
    format_probe_prompt,
    isolated_inference,
)
from lib.config import TrainingConfig
from lib.dataset import format_alpaca_prompts, format_chatml_prompts
from lib.engine import run_sft_training


class ProbeTokenizer:
    eos_token_id = 2
    pad_token_id = None
    padding_side = "right"

    def __init__(self):
        self.prompts = []
        self.responses = [
            "async fn in traits",
            "Rust 1.70.",
            "TryInto, TryFrom, FromIterator",
            "No. Requires Rust 1.80.",
        ]

    def __call__(self, prompt, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        self.prompts.append(prompt)
        return BatchEncoding({"input_ids": torch.tensor([[5, 6, 7]])})

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return self.responses[(tokens[0] - 10) % 4]


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout()
        self.dropout.eval()
        self.config = SimpleNamespace(use_cache=False)
        self.calls = []
        self.fail = False
        self.truncate = False

    def generate(self, **kwargs):
        assert not self.training
        assert not self.dropout.training
        assert not torch.is_grad_enabled()
        assert self.config.use_cache is True
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        assert kwargs["use_cache"] is True
        assert kwargs["pad_token_id"] == 2
        random.random()
        np.random.random()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("generation failed")
        token = 10 + len(self.calls) % 4
        self.calls.append(kwargs)
        if self.truncate:
            return torch.tensor([[5, 6, 7] + [token] * MAX_NEW_TOKENS])
        return torch.tensor([[5, 6, 7, token, 2]])


@pytest.fixture
def setup_probe(tmp_path):
    config = TrainingConfig(model_path="local/model", output_dir=str(tmp_path))
    tokenizer = ProbeTokenizer()
    model = ProbeModel()
    accelerator = MagicMock(distributed_type="NO")
    accelerator.unwrap_model.return_value = model
    callback = CheckpointBenchmarkCallback(
        config,
        tokenizer,
        format_chatml_prompts,
        accelerator,
        logging.getLogger("probe-test"),
    )
    args = SimpleNamespace(output_dir=str(tmp_path / "custom-checkpoints"))
    state = TrainerState(global_step=0, epoch=0, is_world_process_zero=True)
    return callback, args, state, model, tokenizer, accelerator


def rng_values():
    return random.random(), np.random.random(), torch.rand(3)


def assert_values_equal(left, right):
    assert left[0] == right[0]
    assert left[1] == right[1]
    assert torch.equal(left[2], right[2])


def test_baseline_and_every_save_append_history(setup_probe):
    callback, args, state, model, tokenizer, accelerator = setup_probe
    control = TrainerControl()
    assert callback.on_train_begin(args, state, control, model=model) is control
    for step in (50, 100):
        state.global_step = step
        state.epoch = step / 200
        assert callback.on_save(args, state, control, model=model) is control
    rows = [json.loads(line) for line in callback.history_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["baseline", "checkpoint", "checkpoint"]
    assert [row["step"] for row in rows] == [0, 50, 100]
    assert rows[-1]["epoch"] == 0.5
    assert rows[0]["checkpoint_path"] is None
    assert rows[-1]["checkpoint_path"] == str(Path(args.output_dir) / "checkpoint-100")
    assert callback.history_path.parent != Path(args.output_dir)
    assert len({row["run_id"] for row in rows}) == 1
    assert all(row["summary"]["rubric_pass_count"] == 4 for row in rows)
    assert all(row["benchmark_ids"] == list(PROBE_IDS) for row in rows)
    assert len(model.calls) == len(tokenizer.prompts) == 12
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token_id is None
    assert accelerator.wait_for_everyone.call_count == 6
    assert rows[0]["results"][0]["finish_reason"] == "eos"
    assert rows[0]["results"][0]["generated_tokens"] == 2
    assert rows[0]["results"][0]["response"] == "async fn in traits"
    assert rows[0]["results"][0]["prompt"] == tokenizer.prompts[0]


@pytest.mark.parametrize("fail", [False, True])
def test_modes_caches_and_rng_restored_even_on_failure(setup_probe, fail):
    callback, args, state, model, _, _ = setup_probe
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = rng_values()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            callback.on_save(args, state, TrainerControl(), model=model)
        assert not callback.history_path.exists()
    else:
        callback.on_save(args, state, TrainerControl(), model=model)
    assert_values_equal(rng_values(), expected)
    assert model.training is True
    assert model.dropout.training is False
    assert model.config.use_cache is False
    assert model.weight.grad is None


def test_inference_preserves_initial_eval_and_mps_state():
    model = ProbeModel().eval()
    model.config.use_cache = True
    rng = torch.tensor([1, 2], dtype=torch.uint8)
    with (
        patch("torch.mps.get_rng_state", return_value=rng),
        patch("torch.mps.set_rng_state") as restore,
    ):
        with isolated_inference(model, torch.device("mps")):
            assert not model.training
        restore.assert_called_once_with(rng)
    assert not model.training
    assert model.config.use_cache is True


def test_resume_baseline_appends_and_keeps_old_run(setup_probe):
    callback, args, state, model, tokenizer, accelerator = setup_probe
    callback.on_train_begin(args, state, TrainerControl(), model=model)
    resumed = CheckpointBenchmarkCallback(
        callback.config,
        tokenizer,
        format_chatml_prompts,
        accelerator,
        callback.logger,
    )
    state.global_step = 100
    state.epoch = 0.5
    resumed.on_train_begin(args, state, TrainerControl(), model=model)
    rows = [json.loads(line) for line in callback.history_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["event"] == "resume_baseline"
    assert rows[1]["step"] == 100
    assert rows[0]["run_id"] != rows[1]["run_id"]


def test_worker_rank_waits_without_generating(setup_probe):
    callback, args, state, model, _, accelerator = setup_probe
    state.is_world_process_zero = False
    with patch("lib.checkpoint_benchmark.broadcast_object_list") as broadcast:
        broadcast.side_effect = lambda value, **kwargs: value.__setitem__(0, True)
        callback.on_save(args, state, TrainerControl(), model=model)
    assert not callback.history_path.exists()
    assert not model.calls
    accelerator.unwrap_model.assert_not_called()
    assert accelerator.wait_for_everyone.call_count == 2


def test_worker_fails_when_rank_zero_fails(setup_probe):
    callback, args, state, model, _, accelerator = setup_probe
    state.is_world_process_zero = False
    with pytest.raises(RuntimeError, match="rank zero"):
        callback.on_save(args, state, TrainerControl(), model=model)
    assert accelerator.wait_for_everyone.call_count == 2


def test_truncation_prevents_pass(setup_probe):
    callback, args, state, model, _, _ = setup_probe
    model.truncate = True
    callback.on_save(args, state, TrainerControl(), model=model)
    report = json.loads(callback.history_path.read_text())
    assert report["summary"]["rubric_pass_count"] == 0
    assert all("output_token_limit" in row["review_reasons"] for row in report["results"])
    assert all(row["finish_reason"] == "length" for row in report["results"])


def test_write_errors_are_not_silenced(setup_probe):
    callback, args, state, model, _, _ = setup_probe
    callback.history_path.mkdir()
    with pytest.raises(IsADirectoryError):
        callback.on_save(args, state, TrainerControl(), model=model)
    assert model.training and not model.config.use_cache


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED", "TP", "XLA"])
def test_unsupported_distributed_strategies_fail_explicitly(setup_probe, distributed_type):
    callback, _, _, _, tokenizer, accelerator = setup_probe
    accelerator.distributed_type = distributed_type
    with pytest.raises(ValueError, match="no-checkpoint-benchmark"):
        CheckpointBenchmarkCallback(
            callback.config,
            tokenizer,
            format_chatml_prompts,
            accelerator,
            callback.logger,
        )


def test_missing_probe_ids_or_eos_fail(setup_probe):
    callback, _, _, _, tokenizer, accelerator = setup_probe
    with patch("lib.checkpoint_benchmark.load_benchmark", return_value=[]):
        with pytest.raises(ValueError, match="missing probe IDs"):
            CheckpointBenchmarkCallback(
                callback.config,
                tokenizer,
                format_chatml_prompts,
                accelerator,
                callback.logger,
            )
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        CheckpointBenchmarkCallback(
            callback.config,
            tokenizer,
            format_chatml_prompts,
            accelerator,
            callback.logger,
        )


def test_prompt_formats_match_the_trainers():
    from train_k2_horizon import format_k2_prompts

    for formatter in (format_chatml_prompts, format_alpaca_prompts, format_k2_prompts):
        prompt = format_probe_prompt("Question?", formatter)
        training = formatter(
            {
                "instruction": ["Question?"],
                "input": [""],
                "output": ["ANSWER"],
            }
        )["text"][0]
        assert training.startswith(prompt + "ANSWER")
        assert "Question?" in prompt
    with pytest.raises(ValueError, match="exactly once"):
        format_probe_prompt("Question?", lambda _: {"text": ["no answer marker"]})


def test_engine_registers_default_callback_with_actual_formatter(tmp_path):
    config = TrainingConfig(model_path="test/model", output_dir=str(tmp_path))
    model, tokenizer, trainer = MagicMock(), MagicMock(), MagicMock()
    trainer.train.return_value = SimpleNamespace(training_loss=0.5)
    formatter = format_alpaca_prompts
    with (
        patch("lib.engine.SFTTrainer", return_value=trainer),
        patch("lib.engine.CheckpointBenchmarkCallback") as callback,
        patch("lib.engine.build_training_arguments"),
    ):
        run_sft_training(
            model,
            tokenizer,
            [],
            config,
            prompt_formatter_fn=formatter,
        )
    callback.assert_called_once()
    assert callback.call_args.args[:4] == (config, tokenizer, formatter, trainer.accelerator)
    trainer.add_callback.assert_called_once_with(callback.return_value)
    assert json.loads((tmp_path / "manifest.json").read_text())["checkpoint_benchmark"] is True


def test_engine_optout_does_not_load_benchmark(tmp_path):
    config = TrainingConfig(
        model_path="test/model",
        output_dir=str(tmp_path),
        checkpoint_benchmark=False,
    )
    trainer = MagicMock()
    trainer.train.return_value = SimpleNamespace(training_loss=0.5)
    with (
        patch("lib.engine.SFTTrainer", return_value=trainer),
        patch("lib.engine.CheckpointBenchmarkCallback") as callback,
        patch("lib.engine.build_training_arguments"),
    ):
        run_sft_training(MagicMock(), MagicMock(), [], config)
    callback.assert_not_called()
    trainer.add_callback.assert_not_called()


def test_real_trainer_saves_probe_history_without_changing_weights(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

    torch.manual_seed(123)
    model_config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    monitored = GPT2LMHeadModel(model_config)
    plain = GPT2LMHeadModel(model_config)
    plain.load_state_dict(monitored.state_dict())
    dataset = [
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},
        {"input_ids": [8, 7, 6, 5], "labels": [8, 7, 6, 5]},
    ]
    for model, name in ((plain, "plain"), (monitored, "monitored")):
        output = tmp_path / name
        args = TrainingArguments(
            output_dir=str(output / "checkpoints"),
            max_steps=2,
            per_device_train_batch_size=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=1,
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            disable_tqdm=True,
            seed=123,
            gradient_checkpointing=True,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        if name == "monitored":
            trainer.add_callback(
                CheckpointBenchmarkCallback(
                    TrainingConfig(model_path="tiny-local", output_dir=str(output)),
                    ProbeTokenizer(),
                    format_chatml_prompts,
                    trainer.accelerator,
                    logging.getLogger("probe-integration"),
                )
            )
        with patch("lib.checkpoint_benchmark.MAX_NEW_TOKENS", 4):
            trainer.train()
    for left, right in zip(plain.parameters(), monitored.parameters(), strict=True):
        assert torch.equal(left, right)
    history = tmp_path / "monitored" / "benchmark_history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert [row["event"] for row in rows] == ["baseline", "checkpoint", "checkpoint"]
    assert not (tmp_path / "monitored" / "checkpoints" / "checkpoint-1").exists()
    assert (tmp_path / "monitored" / "checkpoints" / "checkpoint-2").is_dir()
    assert monitored.config.use_cache is False
