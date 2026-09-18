# IFM K2-Horizon-3.7B LoRA Training & Coding Agent Guide

This document provides complete instructions for training a Rust expert LoRA adapter on [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B), loading the base model and adapter together, and using them with AI coding agents (Cline, Roo Code, Continue, Aider, Cursor, Copilot).

---

## Model Overview

**IFM/K2-Horizon-3.7B** is a dense 3.7B-parameter foundation model created by the Institute of Foundation Models (IFM), purpose-built for reasoning, software engineering, and tool-augmented agentic execution.

- **Parameters:** 3.7B dense (36 layers, hidden dimension 2560, intermediate size 10240, 32 attention heads, 8 KV heads)
- **Context Length:** Native 512K context window (524,288 tokens)
- **Architecture:** `K2HorizonForCausalLM` (`trust_remote_code=True`)
- **Coding Benchmarks:** SWE-bench Verified: **68.6%**, SciCode: **25.9%**, Terminal-Bench 2.1: **25.1%**, BFCL Tool Use: **50.9%**
- **Special Tokens:** `<|ifm|begin_of_text|>`, `<|ifm|endoftext|>`, `<|ifm|im_start|>`, `<|ifm|im_end|>`, `<ifm|think>`

---

## 1. Quick Start: Training

### Step 1: Validate Setup (Dry-Run)

Validate dependencies, CUDA/GPU memory, and dataset integrity before training:

```bash
uv run python train_k2_horizon.py --dry-run
```

This verifies:
- GPU availability and VRAM capacity
- Core dependencies (`transformers`, `peft`, `trl`, `torch`)
- Data format (`data/datasets/train.jsonl` with `instruction`, `input`, `output`)
- Built-in training parameters and hardware profiles

### Step 2: Start Training

Run with the default RTX 3090/4090 profile (bfloat16):

```bash
uv run python train_k2_horizon.py
```

Outputs the trained adapter to `models/k2_horizon_rust_lora`.

---

## 2. Hardware Profiles

Select a profile matching your available GPU hardware using `--hardware-profile`:

| Profile | GPU | VRAM | Batch Size | Grad Accum | Max Seq Len | Mode |
|---|---|---|---|---|---|---|
| `rtx4070` | RTX 4070 / 3060 | 12GB | 1 | 4 | 1024 | 4-bit QLoRA |
| `rtx3090_4090` | RTX 3090 / 4090 | 24GB | 1 | 4 | 2048 | 16-bit bf16 (**Default**) |
| `rtx6000` | RTX 6000 Ada | 48GB | 2 | 2 | 4096 | 16-bit bf16 |
| `a100` | NVIDIA A100 | 40GB | 2 | 2 | 4096 | 16-bit bf16 |
| `a100_80gb` | A100 / H100 | 80GB | 4 | 2 | 8192 | 16-bit bf16 |
| `dev` | Any GPU | Variable | 1 | 1 | 512 | Minimal test profile |

### Examples

```bash
# Consumer GPU (12GB VRAM):
uv run python train_k2_horizon.py --hardware-profile rtx4070

# Workstation GPU (24GB VRAM):
uv run python train_k2_horizon.py --hardware-profile rtx3090_4090

# High-memory cloud GPU:
uv run python train_k2_horizon.py --hardware-profile a100

# Custom parameters:
uv run python train_k2_horizon.py \
  --batch-size 2 \
  --max-seq-length 4096 \
  --num-epochs 3 \
  --learning-rate 1.5e-4
```

---

## 3. Serving the Model for Coding Agents

Once the LoRA adapter is trained, load it together with `IFM/K2-Horizon-3.7B` and connect to your preferred coding agent.

### Option A: Built-in Coding Agent Server (`serve_coding_agent.py`)

The repository includes a ready-to-run OpenAI-compatible server that automatically translates coding agent tool requests and streaming chunks into K2-Horizon's native format:

```bash
# Start server on port 8000
uv run python serve_coding_agent.py \
  --model-path IFM/K2-Horizon-3.7B \
  --lora-path models/k2_horizon_rust_lora \
  --port 8000

# 4-bit quantized mode (for 12GB VRAM GPUs):
uv run python serve_coding_agent.py --load-in-4bit --port 8000

# Test directly from CLI:
uv run python serve_coding_agent.py --test "How do I implement an async WebSocket server in Rust with Tokio and Axum?"
```

The server provides:
- `/v1/chat/completions` (streaming SSE & non-streaming)
- `/v1/models`
- Native tool/function calling extraction (`<ifm|tool_calls>`)
- Reasoning content forwarding (`<ifm|think>` $\to$ `reasoning_content`)

---

### Option B: High-Throughput Serving with vLLM

For high-concurrency production deployments:

```bash
vllm serve IFM/K2-Horizon-3.7B \
  --enable-lora \
  --lora-modules rust-lora=models/k2_horizon_rust_lora \
  --trust-remote-code \
  --dtype bfloat16 \
  --reasoning-parser k2_horizon \
  --enable-auto-tool-choice \
  --tool-call-parser k2_horizon \
  --port 8000
```

---

### Option C: SGLang Serving

Using the validated SGLang recipe:

```bash
sglang serve \
  --model-path IFM/K2-Horizon-3.7B \
  --lora-paths rust-lora=models/k2_horizon_rust_lora \
  --trust-remote-code \
  --dtype bfloat16 \
  --reasoning-parser k2_horizon \
  --host 0.0.0.0 \
  --port 30000
```

---

### Option D: Ollama Export & GGUF

To package and run in Ollama:

1. **Create metadata manifest:**
   ```bash
   uv run python scripts/09_create_modelfile.py \
     --model models/k2_horizon_rust_lora \
     --name rust-k2-horizon \
     --base k2_horizon
   ```

2. **Merge LoRA and create GGUF:**
   ```bash
   uv run python scripts/10_export_ollama.py \
     --model models/k2_horizon_rust_lora \
     --name rust-k2-horizon \
     --base IFM/K2-Horizon-3.7B
   ```

3. **Create and run in Ollama:**
   ```bash
   ollama create rust-k2-horizon -f models/rust-k2-horizon_gguf/Modelfile
   ollama run rust-k2-horizon "Write an embedded Rust driver for I2C sensors using embedded-hal."
   ```

---

## 4. Coding Agent Configurations

Once your server is running at `http://localhost:8000/v1` (or Ollama at `http://localhost:11434/v1`):

### Cline / Roo Code (VS Code Extension)

1. Open Cline/Roo Code settings.
2. Select **OpenAI Compatible** as API Provider.
3. Configure:
   - **Base URL:** `http://localhost:8000/v1`
   - **API Key:** `EMPTY` (or any string)
   - **Model ID:** `k2-horizon-3.7b-rust-lora`
4. Enable **Tools / Function Calling**.

### Continue.dev (VS Code / JetBrains)

Add to `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "K2 Horizon 3.7B (Rust Expert)",
      "provider": "openai",
      "model": "k2-horizon-3.7b-rust-lora",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "EMPTY"
    }
  ]
}
```

### Aider

Launch Aider with:

```bash
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key EMPTY \
      --model openai/k2-horizon-3.7b-rust-lora
```

---

## 5. Native Chat & Tool Format Reference

K2-Horizon uses standard ChatML tags with the IFM namespace:

```text
<|ifm|im_start|>system
# Tools
You may call one or more tools to assist with the user query.

Available tools are:

<ifm|tools>
{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
</ifm|tools>

When calling tools, you MUST follow the tool-call format below:
<ifm|tool_calls>
<ifm|tool_call>{"name": <function-name>, "arguments": <args-json-object>}</ifm|tool_call>
</ifm|tool_calls>
<|ifm|im_end|>
<|ifm|im_start|>user
Inspect src/main.rs and fix lifetime annotations.<|ifm|im_end|>
<|ifm|im_start|>assistant
<ifm|think>
Let's first read src/main.rs using the read_file tool.
</ifm|think>
<ifm|tool_calls>
<ifm|tool_call>{"name": "read_file", "arguments": {"path": "src/main.rs"}}</ifm|tool_call>
</ifm|tool_calls><|ifm|im_end|>
```
