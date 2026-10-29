"""Small, state-isolated development probes using the live training model."""

import gc
import hashlib
import json
import logging
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from accelerate.utils import broadcast_object_list
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import TrainingConfig
from .version_eval import (
    BENCHMARK_FILE,
    RUBRIC_VERSION,
    decode_completion,
    load_benchmark,
    score_item,
    summarize,
)

PROBE_IDS = (
    "diff_1.74_1.75",
    "reverse_oncelock",
    "edition_2021_prelude",
    "msrv_lazylock",
)
MAX_NEW_TOKENS = 256
PromptFormatter = Callable[[dict[str, list[str]]], dict[str, list[str]]]


def _release_inference_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def format_probe_prompt(question: str, formatter: PromptFormatter) -> str:
    marker = "__RUST_LORA_BENCHMARK_ANSWER__"
    text = formatter(
        {
            "instruction": [question],
            "input": [""],
            "output": [marker],
        }
    )["text"][0]
    if text.count(marker) != 1:
        raise ValueError(
            "Training prompt formatter must preserve the assistant answer exactly once"
        )
    return text.split(marker, 1)[0]


@contextmanager
def isolated_inference(model: Any, device: torch.device) -> Iterator[None]:
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    module_modes = [(module, module.training) for module in model.modules()]
    # Unsloth's generate() disables gradient checkpointing and only re-arms it when the model
    # was in training mode on entry; we enter in eval mode, so restore it ourselves or the next
    # training step keeps every activation and exhausts VRAM.
    checkpointing = [
        (module, module.gradient_checkpointing)
        for module in model.modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    checkpointing_mode = next((mode for _, mode in checkpointing if mode), False)
    was_training = model.training
    cache = model.config.use_cache
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            model.eval()
            model.config.use_cache = True
            yield
    finally:
        model.config.use_cache = cache
        if was_training and hasattr(model, "for_training"):
            model.for_training(use_gradient_checkpointing=checkpointing_mode)
        for module, training in module_modes:
            module.training = training
        for module, mode in checkpointing:
            module.gradient_checkpointing = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


class CheckpointBenchmarkCallback(TrainerCallback):
    def __init__(
        self,
        config: TrainingConfig,
        tokenizer: Any,
        formatter: PromptFormatter,
        accelerator: Any,
        logger: logging.Logger,
    ) -> None:
        if str(accelerator.distributed_type).split(".")[-1] not in ("NO", "MULTI_GPU", "MULTI_CPU"):
            raise ValueError(
                "Checkpoint benchmarks support CPU, MPS, CUDA, and ordinary DDP, not sharded training. "
                "Use --no-checkpoint-benchmark for this distributed strategy."
            )
        items = {item["id"]: item for item in load_benchmark()}
        missing = set(PROBE_IDS) - items.keys()
        if missing:
            raise ValueError(f"Checkpoint benchmark is missing probe IDs: {sorted(missing)}")
        self.items = [items[probe_id] for probe_id in PROBE_IDS]
        self.prompts = [format_probe_prompt(item["question"], formatter) for item in self.items]
        if tokenizer.eos_token_id is None:
            raise ValueError("Checkpoint benchmarks require a tokenizer EOS token")
        self.benchmark_sha256 = hashlib.sha256(BENCHMARK_FILE.read_bytes()).hexdigest()
        self.config = config
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.run_id = uuid4().hex
        self.history_path = Path(config.output_dir) / "benchmark_history.jsonl"

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        event = "resume_baseline" if state.global_step else "baseline"
        self._run(event, args, state, kwargs["model"])
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        self._run("checkpoint", args, state, kwargs["model"])
        return control

    def _run(
        self,
        event: str,
        args: TrainingArguments,
        state: TrainerState,
        wrapped_model: Any,
    ) -> None:
        self.accelerator.wait_for_everyone()
        succeeded = [False]
        try:
            if state.is_world_process_zero:
                model = self.accelerator.unwrap_model(wrapped_model)
                self._record(event, args, state, model)
                succeeded[0] = True
        finally:
            broadcast_object_list(succeeded, from_process=0)
            self.accelerator.wait_for_everyone()
            if not succeeded[0] and not state.is_world_process_zero:
                raise RuntimeError("Checkpoint benchmark failed on rank zero; see its error log")

    def _record(
        self,
        event: str,
        args: TrainingArguments,
        state: TrainerState,
        model: Any,
    ) -> None:
        device = next(model.parameters()).device
        results = []
        self.logger.info(
            "Running four checkpoint benchmark probes (%s, step %s)", event, state.global_step
        )
        with isolated_inference(model, device):
            for item, prompt in zip(self.items, self.prompts, strict=True):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(device)
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id,
                )
                completion = decode_completion(
                    self.tokenizer,
                    tokens[0],
                    inputs["input_ids"].shape[1],
                    MAX_NEW_TOKENS,
                )
                result = {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "question_revision": item["question_revision"],
                    "reference_answer": item["reference_answer"],
                    "prompt": prompt,
                    **completion,
                    **score_item(completion["response"], item),
                }
                if completion["finish_reason"] == "length":
                    result["review_reasons"].append("output_token_limit")
                    result["rubric_pass"] = False
                results.append(result)
                del tokens, inputs
        _release_inference_memory(device)
        report = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "step": state.global_step,
            "epoch": state.epoch,
            "checkpoint_path": str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
            if event == "checkpoint"
            else None,
            "model_path": self.config.model_path,
            "benchmark_sha256": self.benchmark_sha256,
            "rubric_version": RUBRIC_VERSION,
            "benchmark_ids": list(PROBE_IDS),
            "generation": {
                "batch_size": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "prompt_format": "training_formatter",
                "device": str(device),
            },
            "summary": summarize(results),
            "results": results,
        }
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        self.logger.info(
            "Checkpoint benchmark %s at step %s: %s/%s rubric passes; history: %s",
            event,
            state.global_step,
            report["summary"]["rubric_pass_count"],
            len(self.items),
            self.history_path,
        )
