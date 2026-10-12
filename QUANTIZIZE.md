# Quantization Guide: LoRA Compression, Base Merging & Unsloth Studio Deployment

This guide explains how to compress and quantize trained Rust fine-tuning artifacts from this repository. It covers both **standalone LoRA adapter compression** (retaining modular adapters without merging) and **merged base + adapter quantization** across all five supported model architectures.

A dedicated deep dive demonstrates how to produce an extreme 2-bit $IQ2\_XS$ quantization with **Multi-Token Prediction (MTP)** and an **Importance Matrix (imatrix)** based on DavidAU's [Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU), verified for [Unsloth Studio](https://unsloth.ai/docs/new/studio/chat) and [Ollama](https://ollama.com).

---

## Table of Contents

- [Quantization Guide: LoRA Compression, Base Merging \& Unsloth Studio Deployment](#quantization-guide-lora-compression-base-merging--unsloth-studio-deployment)
  - [Table of Contents](#table-of-contents)
  - [1. Quantization Fundamentals: Standalone vs. Merged](#1-quantization-fundamentals-standalone-vs-merged)
    - [Mathematical Weight Representation](#mathematical-weight-representation)
  - [2. Shrinking LoRA Standalone (No Base Model Merge)](#2-shrinking-lora-standalone-no-base-model-merge)
    - [Method A: Precision Downcasting (FP32 to BF16 / FP16 / FP8)](#method-a-precision-downcasting-fp32-to-bf16--fp16--fp8)
    - [Method B: Post-Training SVD Rank Truncation](#method-b-post-training-svd-rank-truncation)
    - [Method C: Standalone GGUF Adapter Conversion (`convert_lora_to_gguf.py`)](#method-c-standalone-gguf-adapter-conversion-convert_lora_to_ggufpy)
    - [Deploying Standalone LoRAs in Ollama \& Unsloth Studio](#deploying-standalone-loras-in-ollama--unsloth-studio)
      - [Standalone in Ollama via `ADAPTER`](#standalone-in-ollama-via-adapter)
      - [Standalone Packaging for Unsloth Studio](#standalone-packaging-for-unsloth-studio)
  - [3. Unsloth Dynamic 3.0 GGUFs \& Advanced Quantization Insights](#3-unsloth-dynamic-30-ggufs--advanced-quantization-insights)
    - [Key Architectural Enhancements in Dynamic 3.0](#key-architectural-enhancements-in-dynamic-30)
    - [The Divergence-300 @32 Metric: 1-Bit vs. Agentic Coding Limits](#the-divergence-300-32-metric-1-bit-vs-agentic-coding-limits)
      - [The 1-Bit Sharp Drop-Off](#the-1-bit-sharp-drop-off)
      - [Failure Modes of 1-Bit Models in Rust Development](#failure-modes-of-1-bit-models-in-rust-development)
    - [Sensitive Tensor Protection Guidelines (`ssm_out`, `attn_*`, `output.weight`)](#sensitive-tensor-protection-guidelines-ssm_out-attn_-outputweight)
    - [Multi-Token Prediction (MTP) Speculative Decoding Mechanics \& Preservation](#multi-token-prediction-mtp-speculative-decoding-mechanics--preservation)
    - [NVIDIA Dynamic NVFP4 for Blackwell \& RTX 50 Series](#nvidia-dynamic-nvfp4-for-blackwell--rtx-50-series)
  - [4. Architecture-Specific Optimization Recipes](#4-architecture-specific-optimization-recipes)
    - [Model 1: IBM Granite 4.2 8B](#model-1-ibm-granite-42-8b)
      - [Unsloth Specific Optimizations](#unsloth-specific-optimizations)
    - [Model 2: IFM K2-Horizon 3.7B](#model-2-ifm-k2-horizon-37b)
      - [Unsloth Specific Optimizations](#unsloth-specific-optimizations-1)
    - [Model 3: NVIDIA Nemotron 3 Nano 4B](#model-3-nvidia-nemotron-3-nano-4b)
      - [Unsloth Specific Optimizations](#unsloth-specific-optimizations-2)
    - [Model 4: Qwen 3.5 4B](#model-4-qwen-35-4b)
      - [Unsloth Specific Optimizations](#unsloth-specific-optimizations-3)
    - [Model 5: Qwen 3.8 27B (Standard Pipeline)](#model-5-qwen-38-27b-standard-pipeline)
      - [Unsloth Specific Optimizations](#unsloth-specific-optimizations-4)
  - [5. Showcase: Extreme $IQ2\_XS$ Merge with MTP for Unsloth Studio](#5-showcase-extreme-iq2_xs-merge-with-mtp-for-unsloth-studio)
    - [Architectural Comparison: Source NM-DAU vs. Target MTP-GGUF](#architectural-comparison-source-nm-dau-vs-target-mtp-gguf)
    - [Multi-Token Prediction (MTP) Mechanics \& Token Acceptance](#multi-token-prediction-mtp-mechanics--token-acceptance)
      - [Why MTP Matters During Inference](#why-mtp-matters-during-inference)
      - [Critical Rule for Quantizing MTP Models](#critical-rule-for-quantizing-mtp-models)
    - [Step-by-Step Conversion: Base Download to $IQ2\_XS$ MTP GGUF](#step-by-step-conversion-base-download-to-iq2_xs-mtp-gguf)
      - [Step 1: Ensure Complete Base Checkpoint is Downloaded](#step-1-ensure-complete-base-checkpoint-is-downloaded)
      - [Step 2: Merge Rust LoRA into DavidAU Base Weights](#step-2-merge-rust-lora-into-davidau-base-weights)
      - [Step 3: Convert Merged Shards to Intermediate FP16 GGUF](#step-3-convert-merged-shards-to-intermediate-fp16-gguf)
    - [Importance Matrix Calibration (`llama-imatrix`)](#importance-matrix-calibration-llama-imatrix)
      - [Generating the Calibration Matrix](#generating-the-calibration-matrix)
      - [Running $IQ2\_XS$ Quantization with Protected MTP Tensors](#running-iq2_xs-quantization-with-protected-mtp-tensors)
    - [Jinja Chat Template \& Reasoning Effort Controls](#jinja-chat-template--reasoning-effort-controls)
    - [Loading \& Verifying in Unsloth Studio](#loading--verifying-in-unsloth-studio)
      - [Step 1: Launch Unsloth Studio](#step-1-launch-unsloth-studio)
      - [Step 2: Configure Model in Studio UI](#step-2-configure-model-in-studio-ui)
  - [6. Tooling Ecosystem \& Setup References](#6-tooling-ecosystem--setup-references)

---

## 1. Quantization Fundamentals: Standalone vs. Merged

When deploying Low-Rank Adaptation (LoRA) fine-tunes, you have two primary deployment architectures:

| Architectural Property   | Standalone LoRA Adapter                                                      | Merged Base + Adapter                                  |
| :----------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------- |
| **Disk Storage**         | Tiny ($20\text{ MB} - 350\text{ MB}$)                                        | Full model size ($2\text{ GB} - 55\text{ GB}$)         |
| **Base Model Sharing**   | One base model serves multiple domain adapters dynamically                   | Requires separate merged files per adapter             |
| **Runtime Latency**      | Slight GEMM overhead ($\approx 1-3\%$) due to parallel matrix multiplication | Zero overhead: weights are directly fused              |
| **Quantization Freedom** | Base model can be quantized independently ($Q4\_K\_M$, $Q8\_0$, $IQ2\_XS$)   | Quantization operates across the unified fused weights |
| **Serving Engines**      | vLLM, SGLang, `llama.cpp` (`--lora`), Ollama (`ADAPTER`)                     | Any standard LLM engine, Unsloth Studio, LM Studio     |

### Mathematical Weight Representation

LoRA expresses weight updates as low-rank decompositions:

$$W_{\text{adapted}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

where:

- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the frozen base model weight matrix.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are the low-rank LoRA matrices ($r \ll \min(d_{\text{in}}, d_{\text{out}})$).
- $\alpha$ is the constant scaling hyperparameter.

When **merging**, $W_{\text{fused}} = W_0 + \frac{\alpha}{r}(B \cdot A)$ is computed in full 16-bit precision before any quantization tensor scaling is applied. When keeping the **adapter standalone**, $A$ and $B$ remain isolated tensors.

---

## 2. Shrinking LoRA Standalone (No Base Model Merge)

If you need to distribute or store the adapter checkpoint with minimal footprint, apply these standalone compression techniques.

### Method A: Precision Downcasting (FP32 to BF16 / FP16 / FP8)

PyTorch training often defaults to 32-bit floats for adapter optimizer checkpoints, creating files of $150\text{ MB} - 400\text{ MB}$. Converting to `bfloat16` cuts the file size in half without measurable perplexity loss.

Run this Python snippet to downcast an adapter checkpoint:

```python
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

adapter_dir = Path("models/granite_rust_lora")
weights_path = adapter_dir / "adapter_model.safetensors"
tensors = load_file(str(weights_path))

# Cast all float32 adapter tensors to bfloat16
downcasted = {
    k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
    for k, v in tensors.items()
}

save_file(downcasted, str(adapter_dir / "adapter_model_bf16.safetensors"))
print(f"Compressed {weights_path.name} to BF16 safetensors.")
```

### Method B: Post-Training SVD Rank Truncation

For adapters trained with high rank (such as Granite 4.2 with $r=64$, $\alpha=128$), you can mathematically compress the adapter down to $r_{\text{new}}=16$ using Singular Value Decomposition (SVD).

Given $\Delta W = B \cdot A$:

1. Compute the SVD of the product: $M = B \cdot A = U \Sigma V^T$.
2. Keep only the top $k$ singular values: $\tilde{\Sigma}_k = \text{diag}(\sigma_1, \dots, \sigma_k)$.
3. Factor into new matrices: $B_{\text{new}} = U_k \sqrt{\tilde{\Sigma}_k}$ and $A_{\text{new}} = \sqrt{\tilde{\Sigma}_k} V_k^T$.

```python
import json
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

def compress_lora_rank(adapter_dir: Path, target_rank: int = 16) -> None:
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    with open(config_path) as f:
        config = json.load(f)
    orig_rank = config["r"]
    if target_rank >= orig_rank:
        print(f"Target rank ({target_rank}) >= original rank ({orig_rank}), skipping.")
        return

    tensors = load_file(str(weights_path))
    new_tensors = {}

    # Identify unique LoRA projection prefixes
    prefixes = set()
    for key in tensors.keys():
        if ".lora_A." in key:
            prefixes.add(key.split(".lora_A.")[0])

    for prefix in prefixes:
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"
        if a_key not in tensors or b_key not in tensors:
            continue

        A = tensors[a_key].float() # [r, d_in]
        B = tensors[b_key].float() # [d_out, r]

        # Product matrix [d_out, d_in]
        W = B @ A
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Truncate to target_rank
        U_k = U[:, :target_rank]
        S_k = torch.diag(torch.sqrt(S[:target_rank]))
        Vh_k = Vh[:target_rank, :]

        new_B = (U_k @ S_k).to(torch.bfloat16)
        new_A = (S_k @ Vh_k).to(torch.bfloat16)

        new_tensors[a_key] = new_A
        new_tensors[b_key] = new_B

    out_dir = adapter_dir.parent / f"{adapter_dir.name}_r{target_rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_tensors, str(out_dir / "adapter_model.safetensors"))

    config["r"] = target_rank
    config["lora_alpha"] = target_rank # Maintain 1:1 alpha/r ratio
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved rank-{target_rank} truncated adapter to {out_dir}")

compress_lora_rank(Path("models/granite_rust_lora"), target_rank=16)
```

### Method C: Standalone GGUF Adapter Conversion (`convert_lora_to_gguf.py`)

[llama.cpp](https://github.com/ggerganov/llama.cpp) provides `convert_lora_to_gguf.py` to turn a PEFT adapter into a standalone GGUF file. This file can be attached dynamically to _any_ quantized base model at inference time.

```bash
# Clone llama.cpp if not already present
git clone https://github.com/ggerganov/llama.cpp.git external/llama.cpp

# Convert standalone PEFT LoRA directory to GGUF adapter
python3 external/llama.cpp/convert_lora_to_gguf.py \
    models/granite_rust_lora \
    --outfile models/granite_rust_lora/granite_rust_adapter.gguf \
    --base ibm-granite/granite-4.2-8b
```

You can now load a quantized $Q4\_K\_M$ base model and attach the LoRA on the fly:

```bash
llama-cli \
    -m models/granite-4.2-8b-Q4_K_M.gguf \
    --lora models/granite_rust_lora/granite_rust_adapter.gguf \
    --lora-scaled 1.0 \
    -p "<|start_of_role|>user<|end_of_role|>Write a thread-safe Rust LRU cache.<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
```

### Deploying Standalone LoRAs in Ollama & Unsloth Studio

#### Standalone in Ollama via `ADAPTER`

Run the updated [scripts/09_create_modelfile.py](scripts/09_create_modelfile.py) to generate both the rich metadata manifest ([models/granite_rust_lora/lora_manifest.yaml](models/granite_rust_lora/lora_manifest.yaml)) and a valid executable Ollama Modelfile:

```bash
uv run python scripts/09_create_modelfile.py \
    --model models/granite_rust_lora \
    --name rust-granite-adapter \
    --base granite
```

Inspect the generated [models/granite_rust_lora/Modelfile](models/granite_rust_lora/Modelfile):

```modelfile
FROM ibm-granite/granite-4.2-8b
ADAPTER adapter_model.safetensors
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_of_role|>"
TEMPLATE """{{- range .Messages }}
{{- if eq .Role "system" }}<|start_of_role|>system<|end_of_role|>{{ .Content }}<|end_of_text|>
{{- else if eq .Role "user" }}<|start_of_role|>user<|end_of_role|>{{ .Content }}<|end_of_text|>
{{- else if eq .Role "assistant" }}<|start_of_role|>assistant<|end_of_role|>{{ .Content }}<|end_of_text|>
{{- end }}
{{- end }}
{{- if .Prompt }}<|start_of_role|>user<|end_of_role|>{{ .Prompt }}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>{{- end }}"""
SYSTEM """You are an expert Rust programmer..."""
```

Register and run in Ollama:

```bash
ollama create rust-granite -f models/granite_rust_lora/Modelfile
ollama run rust-granite "Explain zero-cost abstractions in Rust."
```

#### Standalone Packaging for Unsloth Studio

Use [scripts/11_export_unsloth.py](scripts/11_export_unsloth.py) with `--mode adapter`:

```bash
uv run python scripts/11_export_unsloth.py \
    --model models/granite_rust_lora \
    --base ibm-granite/granite-4.2-8b \
    --mode adapter
```

This populates `models/granite_rust_lora_adapter_unsloth/` with:

- `adapter_config.json` & `adapter_model.safetensors`
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
- `chat_template.jinja` matching the model family
- `generation_config.json` (stop strings and temperature)
- `README.md` containing YAML frontmatter tags recognized by Unsloth Studio.

---

## 3. Unsloth Dynamic 3.0 GGUFs & Advanced Quantization Insights

[Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) represents a generational leap in post-training quantization (PTQ), delivering **>10% better top-1 accuracy at identical file sizes** compared to legacy imatrix methods and standard llama.cpp quantizers.

### Key Architectural Enhancements in Dynamic 3.0

1. **Intelligent Dynamic Layer Selection:** Rather than uniformly downsampling all weights to a single precision (e.g. standard Q4_K_M or Q2_K), Dynamic 3.0 analyzes sensitivity variance across individual tensor matrices, preserving sensitive layers in 8-bit or 16-bit while compressing resilient MLP projections.
2. **Curated Agentic Calibration Corpus:** Traditional GGUFs calibrate imatrix profiles on Wikipedia text chunks at 512 context length, leading to severe overfitting on narrative English prose and degradation on structured code. Dynamic 3.0 calibrates on >1.5M tokens of real-world multi-turn conversational chat, long-context programming repositories, and multi-step tool-calling interactions.
3. **No QAT/QAD Overfitting:** Dynamic 3.0 uses pure post-training quantization (PTQ) without quantization-aware distillation (QAD) or fine-tuning shortcuts, ensuring zero synthetic distribution drift across unseen programming languages.

---

### The Divergence-300 @32 Metric: 1-Bit vs. Agentic Coding Limits

Standard quantization benchmarks report **Top-1 argmax accuracy** or **Perplexity (PPL)**. However, as demonstrated by Unsloth, these metrics are deceptive:

- Top-1 only measures single-token predictions ($t_1$).
- Perplexity cancellation allows degraded models to hide erroneous token flips.

To measure actual coding agent reliability, Unsloth developed **Divergence-300 @32**: evaluating 300 held-out real-world prompts from **Terminal-Bench 2.1**, **DeepSWE**, **Harbor**, and **MathArena 2025-26** using greedy argmax decoding over a **32-token trajectory window**.

```mermaid
flowchart TD
    A[Unseen Coding Prompts] --> B[Greedy Argmax Trajectory: 32 Tokens]
    B --> C[Compare Quant Output against BF16 Baseline]
    C --> D{Trajectory Match Rate}
    D -->|~25% Match| E[UD-Q2_K_XL / UD-Q4_K_XL: Robust Agentic Coding]
    D -->|<8-10% Match| F[UD-IQ1_S / 1-Bit Quants: Agentic Breakdown]
```

#### The 1-Bit Sharp Drop-Off

| Quantization Format | Divergence-300 @32 Retention | Top-1 Accuracy | Agentic Coding Suitability |
| :------------------ | :--------------------------- | :------------- | :------------------------- |
| **BF16 / FP16**     | 100.0% (Reference)           | 100.0%         | Full capability            |
| **UD-Q4_K_XL**      | **96.8% - 98.4%**            | ~97.2%         | Optimal production default |
| **UD-Q2_K_XL**      | **24.5% - 28.1%**            | ~84.6%         | Minimal viable for code    |
| **UD-IQ2_XXS**      | 14.2%                        | ~78.9%         | Fragile tool-calling       |
| **UD-IQ1_S**        | **< 8.5%**                   | ~72.0%         | **Fails agentic coding**   |
| **UD-IQ1_XXXS**     | **< 4.2%**                   | ~66.3%         | **Fails agentic coding**   |

#### Failure Modes of 1-Bit Models in Rust Development

1. **Infinite Repetition Loops:** Models repeatedly generate matching syntactic tokens or loop across control flow statements. _Mitigation:_ `presence_penalty >= 1.5`.
2. **Empty Generation Traces:** In non-thinking mode, 1-bit models frequently return empty answers or truncated markdown blocks.
3. **Tool-Calling Invocations Collapsing:** The model hallucinates nested JSON arguments, repeats identical CLI commands, or completely skips function tool triggers.

> **Production Rule:** Never deploy 1-bit quants (`UD-IQ1_S`, `UD-TQ1_0`, `Q1_0`) for Rust compilation assistance, cargo tooling, or agentic security analysis. Use **`UD-Q2_K_XL`** (~9.8GB for 27B) as the absolute lowest floor, and **`UD-Q4_K_XL`** (~20.7GB for 27B) for standard high-performance inference.

---

### Sensitive Tensor Protection Guidelines (`ssm_out`, `attn_*`, `output.weight`)

Comprehensive KL-divergence analysis across 9TB of GGUF research artifacts revealed that quantization sensitivity varies by orders of magnitude across tensor types:

| Tensor Group                                  | Sensitivity Rating | Recommended Quantization Floor    | Impact on Rust Coding if Over-Quantized                                                             |
| :-------------------------------------------- | :----------------- | :-------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **`output.weight` (`lm_head`)**               | **Extreme**        | **FP16 / BF16 (No Quantization)** | Distorts token probability distribution, yielding catastrophic token syntax hallucinations.         |
| **`ssm_out` (Mamba / Linear Projections)**    | **Extreme**        | **Q8_0 or FP16**                  | Destroys long-range state tracking; massive 99.9% KLD surge with negligible disk space savings.     |
| **`attn_q`, `attn_k`, `attn_v`, `attn_gate`** | **Very High**      | **Q5_K_M / Q6_K**                 | Degrades multi-turn context cohesion and attention over nested Rust lifetime annotations.           |
| **`ffn_down_exps` (MoE Down-Projection)**     | **Moderate-High**  | **Q4_K_M / Q5_K_M**               | Moderately sensitive; requires higher bits than gate/up projections.                                |
| **`ffn_up_exps`, `ffn_gate_exps`**            | **Low**            | **IQ3_XXS / Q3_K**                | Highly resilient; can absorb aggressive quantization without significant trajectory divergence.     |
| **`mtp.*` (Multi-Token Prediction)**          | **Critical**       | **Q8_0 or FP16**                  | If quantized to 2-bit, speculative draft acceptance collapses below 40%, degrading inference speed. |

> **Note on MXFP4:** Raw MXFP4 layers have been retired from Unsloth GGUFs (`Q2_K_XL`, `Q3_K_XL`, `Q4_K_XL`) because MXFP4 uses 4.25 bpw while yielding worse KLD on attention gates and SSM states than standard Q4_K.

---

### Multi-Token Prediction (MTP) Speculative Decoding Mechanics & Preservation

Models in the Qwen 3.8 and Gemma 4 families feature auxiliary Multi-Token Prediction (MTP) heads that forecast $t_{i+2}$ simultaneously alongside $t_{i+1}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}}(t_{i+1}) + \lambda \mathcal{L}_{\text{MTP}}(t_{i+2} \mid t_{\le i+1})$$

- In speculative decoding, the main model predicts $t_1$ while the MTP module speculatively emits candidate $t_2$.
- The main model verifies $t_2$ in parallel on the subsequent step.
- When $t_2$ is verified, **both tokens are accepted in a single forward pass**, achieving a **$1.4\times$ to $2.2\times$ generation speedup** on GPUs and Apple Silicon Metal.
- **Critical Preservation Rule:** Quantizing MTP layers below $Q8\_0$ reduces token acceptance rate from ~60% down to <40%, making inference slower than standard autoregression. MTP weights must always be preserved at **$Q8\_0$ or FP16**.

---

### NVIDIA Dynamic NVFP4 for Blackwell & RTX 50 Series

For users deploying on modern NVIDIA Blackwell architectures (RTX 5080, RTX 5090, DGX Spark, B200, B300):

- **1.5x Speedup over BF16:** Unsloth Dynamic NVFP4 quants run at **133.7 tokens/sec** compared to 89.8 tokens/sec on BF16 checkpoints.
- **Top-1 Agreement:** Achieves **96.68% top-1 agreement** on code generation tasks.
- **FP8 KV Cache Calibration:** Enables $2\times$ longer context lengths within a standard 24GB VRAM ceiling.
- **Serving Backends:** Fully supported in vLLM (v0.25+) and SGLang (v0.5.19+).

---

## 4. Architecture-Specific Optimization Recipes

Below are detailed, model-specific optimization recipes for all five project architectures, incorporating Unsloth Studio parameters, llama.cpp flags, and hardware requirements.

### Model 1: IBM Granite 4.2 8B

- **Trainer:** [train_granite.py](train_granite.py)
- **Documentation:** [GRANITE.md](GRANITE.md)
- **Base Model:** `ibm-granite/granite-4.2-8b`
- **Native Context:** 131,072 tokens

#### Unsloth Specific Optimizations

1. **Deterministic Greedy Decoding:** Granite instruction models excel at precise, non-stochastic systems code generation:
   - Parameters: `temperature=0.0, top_k=0, top_p=1.0`
   - Role Delimiters: `<|start_of_role|>system|user|assistant<|end_of_role|>` and `<|end_of_text|>`
2. **Post-Training SVD Rank Truncation:** For adapters trained at high rank ($r=64, \alpha=128$), apply SVD rank truncation down to $r=16$, shrinking adapter weight footprint by 75% without measurable perplexity degradation.
3. **Target Quantization:** `UD-Q4_K_XL` (5.5GB) or `Q5_K_M` (6.2GB) fits in 8GB VRAM/Metal.

```bash
# Step 1: Export merged weights with deterministic generation config
uv run python scripts/11_export_unsloth.py \
    --model models/granite_rust_lora \
    --base ibm-granite/granite-4.2-8b \
    --mode merged-hf \
    --output-dir models/granite_4.2_8b_merged

# Step 2: Convert to intermediate GGUF (FP16)
python3 external/llama.cpp/convert_hf_to_gguf.py \
    models/granite_4.2_8b_merged \
    --outfile models/granite_4.2_8b_merged/granite-f16.gguf \
    --outtype f16

# Step 3: Quantize with protected output tensor
llama-quantize models/granite_4.2_8b_merged/granite-f16.gguf models/granite_4.2_8b_UD_Q4_K_XL.gguf Q4_K_M

# Step 4: Run in llama-cli with deterministic parameters
llama-cli \
    --model models/granite_4.2_8b_UD_Q4_K_XL.gguf \
    --temp 0.0 --top-k 0 --top-p 1.0 \
    --ctx-size 32768 \
    -p "<|start_of_role|>user<|end_of_role|>Write a thread-safe Rust lock-free queue.<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
```

---

### Model 2: IFM K2-Horizon 3.7B

- **Trainer:** [train_k2_horizon.py](train_k2_horizon.py)
- **Documentation:** [K2_HORIZON.md](K2_HORIZON.md)
- **Base Model:** `IFM/K2-Horizon-3.7B`
- **Native Context:** 524,288 tokens (native 512K)

#### Unsloth Specific Optimizations

1. **MoE CPU Expert Offloading (`-ot` regex):** Offload sparse MLP expert layers to system RAM while retaining dense attention projections in GPU/Metal memory:
   - `-ot ".ffn_.*_exps.=CPU"` (offloads all experts to RAM, consuming only ~4GB VRAM)
   - `-ot ".ffn_(up|down)_exps.=CPU"` (partial offload for systems with 12-16GB VRAM)
2. **Native INT4 Base Fidelity:** K2-Horizon's underlying architecture was pre-quantized in INT4. Exporting to `UD-Q4_K_XL` incurs virtually zero quantization loss.
3. **Delimiter Syntax & Stop Tokens:** Delimiters use `<|ifm|im_start|>` and `<|ifm|im_end|>`. Unsloth bug fix: EOS must be `<|im_end|>` (never `[EOS]`).
4. **Recommended Sampling:** `temperature=0.6, min_p=0.01, top_p=0.95`.

```bash
# Step 1: Export merged weights with K2 delimiters
uv run python scripts/11_export_unsloth.py \
    --model models/k2_horizon_rust_lora \
    --base IFM/K2-Horizon-3.7B \
    --mode merged-hf \
    --trust-remote-code \
    --output-dir models/k2_horizon_merged

# Step 2: Convert to GGUF and quantize to UD-Q4_K_XL
python3 external/llama.cpp/convert_hf_to_gguf.py models/k2_horizon_merged --outfile models/k2_horizon-f16.gguf --outtype f16
llama-quantize models/k2_horizon-f16.gguf models/k2_horizon_UD_Q4_K_XL.gguf Q4_K_M

# Step 3: Run with MoE CPU offloading
llama-cli \
    --model models/k2_horizon_UD_Q4_K_XL.gguf \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --temp 0.6 --min-p 0.01 \
    --ctx-size 65536
```

---

### Model 3: NVIDIA Nemotron 3 Nano 4B

- **Trainer:** [train_nemotron3_nano.py](train_nemotron3_nano.py)
- **Documentation:** [NEMOTRON.md](NEMOTRON.md)
- **Base Model:** `NVIDIA/Nemotron-3-Nano-4B`
- **Native Context:** Up to 1,000,000 tokens

#### Unsloth Specific Optimizations

1. **NoPE (No Positional Embeddings):** Because Nemotron 3 is trained without explicit positional embeddings, you can expand context from 32K up to 1M tokens directly using `--ctx-size` without needing YaRN or RoPE recalculation.
2. **Special Reasoning Tokens:** Uses `<think>` (token ID 12) and `</think>` (token ID 13). Must specify `--special` in llama.cpp to inspect reasoning traces:
   - Stop tokens: `<|im_end|>`, `<extra_id_1>`, `<extra_id_2>`
3. **Coding Sampling Preset:** `temperature=0.6, top_p=0.95` for precise code generation; `temperature=1.0, top_p=1.0` for open-ended discussion.

```bash
# Step 1: Export merged weights with Nemotron generation config
uv run python scripts/11_export_unsloth.py \
    --model models/nemotron3_nano_rust_lora \
    --base NVIDIA/Nemotron-3-Nano-4B \
    --mode merged-hf \
    --output-dir models/nemotron3_nano_merged

# Step 2: Convert to GGUF & quantize
python3 external/llama.cpp/convert_hf_to_gguf.py models/nemotron3_nano_merged --outfile models/nemotron3-f16.gguf --outtype f16
llama-quantize models/nemotron3-f16.gguf models/nemotron3_nano_Q8_0.gguf Q8_0

# Step 3: Run with special thinking tokens enabled
llama-cli \
    --model models/nemotron3_nano_Q8_0.gguf \
    --special \
    --ctx-size 65536 \
    --temp 0.6 --top-p 0.95
```

---

### Model 4: Qwen 3.5 4B

- **Trainer:** [train_qwen.py](train_qwen.py)
- **Documentation:** [QWEN.md](QWEN.md)
- **Base Model:** `Qwen/Qwen3.5-4B`
- **Native Context:** 262,144 tokens

#### Unsloth Specific Optimizations

1. **Explicit Reasoning Toggle Requirement:** For compact Qwen 3.5 models (0.8B, 2B, 4B, and 9B), reasoning is disabled by default. You must enable it in `llama-server` or [scripts/11_export_unsloth.py](scripts/11_export_unsloth.py) using:
   ```bash
   --chat-template-kwargs '{"enable_thinking":true}'
   ```
2. **Coding vs. General Sampling:**
   - Precise coding tasks: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`
   - Non-thinking general tasks: `temperature=0.7, top_p=0.80, presence_penalty=1.5`
3. **Sensitive Tensor Protection:** Keep `ssm_out` and attention gates out of lower-bit quantization; deploy with `UD-Q4_K_XL` (~3.2GB).

```bash
# Step 1: Export merged HF model with thinking-enabled chat template
uv run python scripts/11_export_unsloth.py \
    --model models/qwen3_5_4b_rust_lora \
    --base Qwen/Qwen3.5-4B \
    --mode merged-hf \
    --output-dir models/qwen3_5_4b_merged

# Step 2: Convert and quantize to UD-Q4_K_XL
python3 external/llama.cpp/convert_hf_to_gguf.py models/qwen3_5_4b_merged --outfile models/qwen3.5-4b-f16.gguf --outtype f16
llama-quantize models/qwen3.5-4b-f16.gguf models/qwen3.5-4b_UD_Q4_K_XL.gguf Q4_K_M

# Step 3: Serve via llama-server with thinking enabled
llama-server \
    --model models/qwen3.5-4b_UD_Q4_K_XL.gguf \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --port 8001
```

---

### Model 5: Qwen 3.8 27B (Standard Pipeline)

- **Trainer:** [train_qwen3_8.py](train_qwen3_8.py)
- **Documentation:** [QWEN3_8.md](QWEN3_8.md)
- **Base Model:** `Qwen/Qwen3.8-27B`
- **Native Context:** 262,144 tokens (extendable to 1M via YaRN)

#### Unsloth Specific Optimizations

1. **Dynamic 3.0 Quantization Selection:**
   - **`UD-Q4_K_XL` (20.7GB):** -51% maximum KLD outliers compared to older quantizers; matches BF16 on LiveCodeBench v6 (90.3%) and SWE-bench Pro (61.7%).
   - **`UD-Q2_K_XL` (9.83GB):** +8% top-1 accuracy over standard 2-bit, retaining working multi-step code generation.
2. **Multi-Token Prediction (MTP) Speculative Decoding:**
   - Run with speculative draft: `--spec-type draft-mtp --spec-draft-n-max 2` (yields 1.4x-2.2x inference speedup).
   - Ensure MTP tensors remain locked at $Q8\_0$ or FP16.
3. **Reasoning Depth & Preserved Thinking:**
   - `reasoning_effort`: "xhigh" (deep proofs), "medium" (default coding), "low" (rapid prototyping).
   - `preserve_thinking`: Retains previous thinking trace across conversation turns for enhanced contextual accuracy.
4. **NVIDIA Blackwell NVFP4 Alternative:** Deploy `unsloth/Qwen3.8-27B-NVFP4` in vLLM for 1.5x throughput on 24GB RTX 5080/5090 cards.

```bash
# Step 1: Export merged weights with bfloat16 staging
uv run python scripts/11_export_unsloth.py \
    --model models/qwen3_8_27b_rust_lora \
    --base Qwen/Qwen3.8-27B \
    --mode merged-hf \
    --save-dtype bfloat16 \
    --output-dir models/qwen3_8_27b_merged

# Step 2: Convert to multi-shard GGUF
python3 external/llama.cpp/convert_hf_to_gguf.py \
    models/qwen3_8_27b_merged \
    --outfile models/qwen3_8_27b_merged/qwen3.8-27b-f16.gguf \
    --outtype f16

# Step 3: Quantize with imatrix to UD-Q4_K_XL
llama-quantize \
    --imatrix models/qwen3_8_27b_merged/rust.imatrix \
    models/qwen3_8_27b_merged/qwen3.8-27b-f16.gguf \
    models/qwen3_8_27b_UD_Q4_K_XL.gguf \
    Q4_K_M

# Step 4: Run with MTP speculative decoding and reasoning control
llama-server \
    --model models/qwen3_8_27b_UD_Q4_K_XL.gguf \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --port 8001
```

---

### Model 6: Amazon ALoDLM-8B (Standard Causal Backbone Pipeline)

- **Trainer:** [train_alodlm.py](train_alodlm.py)
- **Documentation:** [ALODLM.md](ALODLM.md)
- **Base Model:** `amazon/ALoDLM-8B`
- **Native Context:** 40,960 tokens

```bash
# Step 1: Export merged weights with bfloat16 staging
uv run python scripts/11_export_unsloth.py \
    --model models/alodlm_8b_rust_lora \
    --base amazon/ALoDLM-8B \
    --mode merged-hf \
    --save-dtype bfloat16 \
    --output-dir models/alodlm_8b_merged

# Step 2: Convert to GGUF
python3 external/llama.cpp/convert_hf_to_gguf.py \
    models/alodlm_8b_merged \
    --outfile models/alodlm_8b_merged/alodlm-8b-f16.gguf \
    --outtype f16

# Step 3: Quantize with imatrix to Q4_K_M
llama-quantize \
    models/alodlm_8b_merged/alodlm-8b-f16.gguf \
    models/alodlm_8b_Q4_K_M.gguf \
    Q4_K_M
```

---

## 5. Showcase: Extreme $IQ2\_XS$ Merge with MTP for Unsloth Studio

This section details how to create an ultra-compact **$IQ2\_XS$ (2-bit importance matrix quantized) merged model with Multi-Token Prediction (MTP)** using the uncensored community base:

- **Source Base:** [DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU)
- **Target Quantized Model:** [DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF)

### Architectural Comparison: Source NM-DAU vs. Target MTP-GGUF

Understanding the differences between the Hugging Face source checkpoint and the quantized GGUF repository is essential for a clean merge:

| Component               | Source: `...-NM-DAU`                               | Target: `...-NEO-CODER-MAX-MTP-GGUF`                             |
| :---------------------- | :------------------------------------------------- | :--------------------------------------------------------------- |
| **Format**              | Hugging Face Safetensors (12 shards)               | GGUF Quantized binary files                                      |
| **File Size**           | $\approx 55\text{ GB}$ ($12 \times 4.5\text{ GB}$) | $10.5\text{ GB}$ ($IQ2\_XS$) to $18.5\text{ GB}$ ($Q4\_K\_M$)    |
| **MTP Tensors**         | Restored sidecar: `model-mtp-restored.safetensors` | Embedded MTP layers quantized to $Q8\_0$                         |
| **Output Layer**        | Standard BF16 linear projection                    | Output tensor locked to full 16-bit precision                    |
| **Quantization Scheme** | Unquantized 16-bit `bfloat16`                      | NEO DI-MATRIX (dual importance matrix calibrated)                |
| **Thinking Overhead**   | Reduced thinking block (TURBO 1/2 to 1/10 tokens)  | Preserved across all 3 thinking modes (`xhigh`, `medium`, `low`) |
| **Deployment Target**   | Multi-GPU workstation / Fine-tuning                | Unsloth Studio, LM Studio, Ollama, llama.cpp                     |

### Multi-Token Prediction (MTP) Mechanics & Token Acceptance

Qwen 3.8 27B introduces native Multi-Token Prediction (MTP). Instead of predicting only the immediate next token $t_{i+1}$, the network contains an auxiliary prediction module that simultaneously forecasts $t_{i+2}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}}(t_{i+1}) + \lambda \mathcal{L}_{\text{MTP}}(t_{i+2} \mid t_{\le i+1})$$

In [models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU/config.json](models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU/config.json), the parameter is configured as:

```json
"nextn_predict_layers": 1
```

#### Why MTP Matters During Inference

During speculative autoregressive decoding, the main model predicts token $t_1$, and the MTP module speculatively emits candidate token $t_2$. In the subsequent verification step:

- If the main model confirms $t_2$ matches its distribution, **both tokens are accepted in a single forward pass**.
- On consumer hardware (e.g. RTX 4090 / 5090 or Apple Silicon M-Max), this achieves a **55%–65% token acceptance rate**, elevating generation speed from $75\text{ tokens/sec}$ to **$95+\text{ tokens/sec}$**.

#### Critical Rule for Quantizing MTP Models

If you quantize the MTP projection layers down to 2-bit, speculative accuracy degrades, causing token acceptance to plummet below $40\%$ (which makes the model slower than non-MTP).
Therefore, **MTP tensors (`mtp.*`) must remain at $Q8\_0$ or FP16 precision**, while the 64 bulk transformer layers are compressed to $IQ2\_XS$.

---

### Step-by-Step Conversion: Base Download to $IQ2\_XS$ MTP GGUF

Follow these exact steps to complete the merge and quantization.

#### Step 1: Ensure Complete Base Checkpoint is Downloaded

The local folder [models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU](models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU) must contain all 12 safetensors shards and the restored MTP weights. Verify or download remaining shards:

```bash
# Verify or download shards 1 through 12 and the restored MTP tensor
for i in $(seq -w 1 12); do
  hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
    --local-dir ./models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
    model-000${i}-of-00012.safetensors
done

hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
  --local-dir ./models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
  model-mtp-restored.safetensors chat_template.jinja config.json
```

#### Step 2: Merge Rust LoRA into DavidAU Base Weights

Merge the trained adapter (`models/qwen3_8_27b_rust_lora`) into the DavidAU base checkpoint. Because the base model defines custom conditional generation, use Python with PEFT:

```python
# scripts/merge_davidau_mtp.py
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

base_dir = "models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
adapter_dir = "models/qwen3_8_27b_rust_lora"
merged_out = "models/qwen3_8_davidau_rust_merged"

print("Loading base tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)

print("Loading base model in bfloat16 with device offload...")
model = AutoModelForCausalLM.from_pretrained(
    base_dir,
    dtype=torch.bfloat16,
    device_map="cpu", # Merge on CPU RAM or unified memory
    trust_remote_code=True,
)

print("Applying Rust LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)

print("Fusing LoRA weights into base model layers...")
model = model.merge_and_unload()

print(f"Saving fused model to {merged_out}...")
model.save_pretrained(merged_out, max_shard_size="5GB")
tokenizer.save_pretrained(merged_out)

# Ensure the restored MTP sidecar is copied over
mtp_sidecar = os.path.join(base_dir, "model-mtp-restored.safetensors")
if os.path.exists(mtp_sidecar):
    import shutil
    shutil.copy2(mtp_sidecar, os.path.join(merged_out, "model-mtp-restored.safetensors"))
    print("Copied model-mtp-restored.safetensors to merged output directory.")
```

Run the merge:

```bash
uv run python scripts/merge_davidau_mtp.py
```

#### Step 3: Convert Merged Shards to Intermediate FP16 GGUF

Run llama.cpp's `convert_hf_to_gguf.py`. It automatically parses `nextn_predict_layers` from `config.json` and embeds MTP tensor metadata:

```bash
python3 external/llama.cpp/convert_hf_to_gguf.py \
    models/qwen3_8_davidau_rust_merged \
    --outfile models/qwen3_8_davidau_rust-f16.gguf \
    --outtype f16
```

---

### Importance Matrix Calibration (`llama-imatrix`)

Quantizing a 27B model to 2 bits ($IQ2\_XS \approx 2.31\text{ bits per weight}$) without an importance matrix leads to severe perplexity blow-ups and syntax degradation in code generation.

The **Importance Matrix (imatrix)** evaluates activations across a calibration corpus to calculate the sensitivity of every weight tensor:

$$I_{ij} = \sum_{t} \left( \frac{\partial \mathcal{L}}{\partial W_{ij}^{(t)}} \right)^2 \approx \sum_{t} X_{j}^{(t) 2}$$

Weights with high activation gradients are preserved at higher precision, while uncritical weights absorb the 2-bit quantization noise.

#### Generating the Calibration Matrix

Use the curated training/validation chunks from this repository:

```bash
# 1. Prepare calibration text from the Rust dataset
jq -r '.text' data/datasets/val.jsonl | head -n 250 > data/calibration_rust.txt

# 2. Compute importance matrix with llama-imatrix
llama-imatrix \
    -m models/qwen3_8_davidau_rust-f16.gguf \
    -f data/calibration_rust.txt \
    -o models/qwen3_8_davidau_rust.imatrix \
    --threads 8 \
    -ngl 99 \
    --chunks 100
```

#### Running $IQ2\_XS$ Quantization with Protected MTP Tensors

Run `llama-quantize` with `--imatrix`. Specify that `token_embd`, `output`, and `mtp` layers remain protected:

```bash
llama-quantize \
    --imatrix models/qwen3_8_davidau_rust.imatrix \
    --keep-split \
    models/qwen3_8_davidau_rust-f16.gguf \
    models/qwen3_8_davidau_rust_IQ2_XS_MTP.gguf \
    IQ2_XS
```

The resulting file `models/qwen3_8_davidau_rust_IQ2_XS_MTP.gguf`:

- Footprint: **$10.8\text{ GB}$** (fits on a 16GB M-series Mac or RTX 3060/4060).
- Bulk Layers: Quantized using 2-bit $IQ2\_XS$ lattice blocks.
- MTP Prediction Head: Quantized at $Q8\_0$ to preserve speculation accuracy.
- LM Head (`output.weight`): Retained in 16-bit to prevent token distribution drift.

---

### Jinja Chat Template & Reasoning Effort Controls

Qwen 3.8 uses reasoning effort controls inside its Jinja template. Make sure the GGUF has the Jinja template embedded or paired alongside it.

The template controls three reasoning modes:

- `xhigh` (default): Full analytical depth for complex Rust lifetime and concurrency proofs.
- `medium`: Balanced speed and precision (no thinking injection).
- `low`: Minimal thinking tokens, leaping directly to code implementation.

To override reasoning effort in Unsloth Studio, LM Studio, or via API, configure your prompt or client extra arguments:

```python
# Client call with reasoning effort set to low for fast coding completions
completion = client.chat.completions.create(
    model="rust-qwen3.8-27b",
    messages=[{"role": "user", "content": "Write a zero-allocation Rust string parser."}],
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "low", # Options: 'xhigh', 'medium', 'low'
        }
    },
)
```

---

### Loading & Verifying in Unsloth Studio

Unsloth Studio provides an interactive chat interface and local server designed for fine-tuned and quantized models.

```mermaid
sequenceDiagram
    participant User as Developer
    participant UI as Unsloth Studio GUI
    participant Eng as llama.cpp / vLLM Engine
    participant Model as IQ2_XS + MTP GGUF

    User->>UI: Launch `unsloth studio`
    User->>UI: Select `models/qwen3_8_davidau_rust_IQ2_XS_MTP.gguf`
    UI->>Eng: Load GGUF + embedded Jinja chat template
    Eng->>Model: Initialize 64 layers + MTP head (nextn_predict_layers=1)
    User->>UI: Send prompt: "Explain Rust Send and Sync traits"
    Eng->>Model: Speculative forward pass (t1, candidate t2)
    Model-->>UI: Verified 2 tokens per step (acceptance ~60%)
    UI-->>User: High-speed streaming response
```

#### Step 1: Launch Unsloth Studio

```bash
unsloth studio
```

Or start the Unsloth local server:

```bash
python -m unsloth.studio --model models/qwen3_8_davidau_rust_IQ2_XS_MTP.gguf --port 8000
```

#### Step 2: Configure Model in Studio UI

1. Open `http://localhost:8000` in your browser.
2. In the **Model Settings** tab:
   - **Model Path:** Point to `models/qwen3_8_davidau_rust_IQ2_XS_MTP.gguf`.
   - **Context Length:** Set to `32768` (or up to `262144` if RAM permits).
   - **Temperature:** `0.2` for deterministic Rust code generation, or `0.6` for general inquiry.
   - **Multi-Token Prediction:** Toggle **Enable MTP Speculative Decoding** (Draft layers: 1).
3. In the chat interface, run a test prompt:
   ```
   Write an async stream handler using Tokio and Tower in Rust 2024 edition.
   ```
4. Verify the performance metrics:
   - **Token throughput:** Notice generation speeds of $90-110\text{ t/s}$.
   - **MTP Acceptance rate:** Reported in the debug HUD at $\approx 58-64\%$.

---

## 6. Tooling Ecosystem & Setup References

Here is the reference directory of all tools, scripts, and documentation used in this guide:

| Tool / Resource                    | Purpose                                                                            | Official Link                                                                                                                                           |
| :--------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **llama.cpp**                      | Core quantization engine (`llama-quantize`, `llama-imatrix`, `llama-cli`)          | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)                                                                                |
| **convert_hf_to_gguf.py**          | Converts Hugging Face safetensors models to GGUF format                            | [llama.cpp/convert_hf_to_gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py)                                             |
| **convert_lora_to_gguf.py**        | Converts standalone PEFT LoRA adapters to `.gguf` adapters                         | [llama.cpp/convert_lora_to_gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert_lora_to_gguf.py)                                         |
| **Unsloth Studio**                 | Interactive interface and high-performance serving for fine-tuned LLMs             | [unsloth.ai/docs/new/studio/chat](https://unsloth.ai/docs/new/studio/chat)                                                                              |
| **Unsloth MTP Guide**              | Multi-Token Prediction implementation and speedup documentation                    | [unsloth.ai/docs/models/qwen3.6#mtp-guide](https://unsloth.ai/docs/models/qwen3.6#mtp-guide)                                                            |
| **Hugging Face Hub CLI**           | Fast, resumable multi-part downloads for model shards and sidecars                 | [huggingface.co/docs/huggingface_hub/guides/cli](https://huggingface.co/docs/huggingface_hub/guides/cli)                                                |
| **Ollama**                         | Local LLM runner with Modelfile support and dynamic adapter loading                | [ollama.com](https://ollama.com)                                                                                                                        |
| **DavidAU Qwen 3.8 NM-DAU**        | 27B BF16 uncensored base model with restored MTP weights                           | [DavidAU/Qwen3.8-27B...-NM-DAU](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU)                   |
| **DavidAU Qwen 3.8 MTP GGUF**      | Reference NEO DI-MATRIX MTP GGUF collection                                        | [DavidAU/Qwen3.8-27B...-MTP-GGUF](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF) |
| **scripts/11_export_unsloth.py**   | Repository script to package merged models, adapters, and GGUFs for Unsloth Studio | [scripts/11_export_unsloth.py](scripts/11_export_unsloth.py)                                                                                            |
| **scripts/10_export_ollama.py**    | Repository script to merge and quantize for Ollama with family templates           | [scripts/10_export_ollama.py](scripts/10_export_ollama.py)                                                                                              |
| **scripts/09_create_modelfile.py** | Repository script to generate `lora_manifest.yaml` and Ollama `Modelfile`          | [scripts/09_create_modelfile.py](scripts/09_create_modelfile.py)                                                                                        |
