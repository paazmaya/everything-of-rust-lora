#!/usr/bin/env python3
"""
OpenAI-Compatible Coding Agent Server for IFM K2-Horizon-3.7B with LoRA

Loads the base model IFM/K2-Horizon-3.7B together with the trained Rust LoRA
adapter and exposes an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).

Features:
- Native joint loading: loads base model and attaches LoRA via PEFT
- Tool/Function Calling: seamlessly bridges K2-Horizon tool format (<ifm|tools>,
  <ifm|tool_calls>) with standard OpenAI tool schemas used by coding agents
  (Cline, Roo Code, Continue, Aider, Cursor, Copilot)
- Reasoning Content: extracts <ifm|think> content into OpenAI reasoning_content
- Streaming & Non-Streaming support: Server-Sent Events (SSE) compatible
- Minimal dependencies: uses standard library HTTP server + PyTorch/Transformers/PEFT

Usage:
    # Run the server on port 8000:
    uv run python serve_coding_agent.py

    # Specify custom LoRA adapter or port:
    uv run python serve_coding_agent.py \
        --model-path IFM/K2-Horizon-3.7B \
        --lora-path models/k2_horizon_rust_lora \
        --port 8000

    # 4-bit quantization for lower VRAM:
    uv run python serve_coding_agent.py --load-in-4bit

    # Direct test query in CLI without running server:
    uv run python serve_coding_agent.py --test "How do I create an async TCP server in Rust with Tokio?"
"""

import argparse
import json
import logging
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("k2-agent-server")

DEFAULT_BASE_MODEL = "IFM/K2-Horizon-3.7B"
DEFAULT_LORA_MODEL = "models/k2_horizon_rust_lora"
SERVED_MODEL_NAME = "k2-horizon-3.7b-rust-lora"
CHUNK_OBJECT_TYPE = "chat.completion.chunk"
CONTENT_TYPE_JSON = "application/json"

# ============================================================================
# Model & Inference Engine
# ============================================================================


class K2HorizonEngine:
    """Inference engine managing K2-Horizon base model and LoRA adapter."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE_MODEL,
        lora_path: str | None = DEFAULT_LORA_MODEL,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        device: str | None = None,
    ):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        logger.info(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.dtype,
        }

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )

        logger.info(f"Loading base model {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

        if lora_path and os.path.exists(lora_path):
            logger.info(f"Attaching LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                is_trainable=False,
            )
            logger.info("LoRA adapter attached successfully.")
        else:
            if lora_path:
                logger.warning(
                    f"LoRA path '{lora_path}' not found on disk. Running base model directly."
                )

        if self.device != "cuda" and not load_in_4bit:
            self.model.to(self.device)

        self.model.eval()
        logger.info("K2-Horizon inference engine ready.")

    def _build_tools_block(self, tools: list[dict[str, Any]]) -> str:
        """Construct the K2-Horizon tool definition block."""
        tool_lines = [
            "# Tools\nYou may call one or more tools to assist with the user query.\n\n",
            "Available tools are:\n\n<ifm|tools>\n",
        ]
        for t in tools:
            tool_lines.append(json.dumps(t) + "\n")
        tool_lines.append("</ifm|tools>\n\n")
        tool_lines.append(
            "When calling tools, you MUST follow the tool-call format below:\n\n"
            "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. "
            "For each call, emit one JSON object with the function name and arguments "
            "on the same line inside <ifm|tool_call></ifm|tool_call> tags:\n\n"
            "<ifm|tool_calls>\n"
            '<ifm|tool_call>{"name": <function-name>, "arguments": <args-json-object>}</ifm|tool_call>\n'
            "</ifm|tool_calls>\n\n"
        )
        return "".join(tool_lines)

    def _render_message(self, msg: dict[str, Any]) -> str:
        """Render a single message into K2-Horizon delimiter syntax."""
        role = msg.get("role")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "user":
            return f"<|ifm|im_start|>user\n{content}<|ifm|im_end|>\n"
        if role == "assistant":
            parts = ["<|ifm|im_start|>assistant\n"]
            if content:
                parts.append(content)
            if tool_calls:
                parts.append("\n<ifm|tool_calls>\n")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    call_obj = {"name": fn.get("name"), "arguments": fn.get("arguments")}
                    parts.append(f"<ifm|tool_call>{json.dumps(call_obj)}</ifm|tool_call>\n")
                parts.append("</ifm|tool_calls>")
            parts.append("<|ifm|im_end|>\n")
            return "".join(parts)
        if role == "tool":
            return f"<|ifm|im_start|>tool\n{content}<|ifm|im_end|>\n"
        return ""

    def format_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Format OpenAI-style messages into K2-Horizon's native chat & tool template.
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

        system_msg = ""
        user_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and not system_msg:
                system_msg = msg.get("content", "")
            else:
                user_messages.append(msg)

        if not system_msg:
            system_msg = (
                "You are an expert Rust programmer, specializing in systems programming, "
                "async Rust, ESP32 embedded development, and the Rust ecosystem."
            )

        prompt_parts: list[str] = ["<|ifm|im_start|>system\n"]
        if tools:
            prompt_parts.append(self._build_tools_block(tools))
        prompt_parts.append(f"{system_msg}<|ifm|im_end|>\n")

        for msg in user_messages:
            rendered = self._render_message(msg)
            if rendered:
                prompt_parts.append(rendered)

        prompt_parts.append("<|ifm|im_start|>assistant\n<ifm|think>\n")
        return "".join(prompt_parts)

    def extract_tools_and_content(self, text: str) -> tuple[str, str | None, list[dict[str, Any]]]:
        """
        Extract thinking content, normal text content, and tool calls from model output.
        """
        reasoning_content = None
        tool_calls: list[dict[str, Any]] = []

        # Extract thinking content
        think_match = re.search(r"<ifm\|think>(.*?)</ifm\|think>", text, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            text = text[: think_match.start()] + text[think_match.end() :]

        # Extract tool calls
        tool_block_match = re.search(r"<ifm\|tool_calls>(.*?)</ifm\|tool_calls>", text, re.DOTALL)
        if tool_block_match:
            block = tool_block_match.group(1)
            text = text[: tool_block_match.start()] + text[tool_block_match.end() :]
            calls = re.findall(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", block, re.DOTALL)
            for c in calls:
                c = c.strip()
                try:
                    call_json = json.loads(c)
                    name = call_json.get("name")
                    args = call_json.get("arguments", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    )
                except Exception as err:
                    logger.warning(f"Failed to parse tool call '{c}': {err}")

        # Clean trailing and stop tokens
        for stop_token in ["<|ifm|im_end|>", "<|ifm|endoftext|>"]:
            text = text.replace(stop_token, "")

        return text.strip(), reasoning_content, tool_calls

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate full completion non-streaming."""
        prompt = self.format_prompt(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, reasoning, tool_calls = self.extract_tools_and_content(raw_text)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(len(new_tokens)),
                "total_tokens": int(inputs["input_ids"].shape[1] + len(new_tokens)),
            },
        }

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ):
        """Yield Server-Sent Events (SSE) for streaming chat completions."""
        prompt = self.format_prompt(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # Yield initial role
        yield {
            "id": chat_id,
            "object": CHUNK_OBJECT_TYPE,
            "created": created,
            "model": SERVED_MODEL_NAME,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        full_raw: list[str] = []
        for text in streamer:
            full_raw.append(text)
            # Filter stop tokens from chunk
            clean_text = text.replace("<|ifm|im_end|>", "").replace("<|ifm|endoftext|>", "")
            if clean_text:
                yield {
                    "id": chat_id,
                    "object": CHUNK_OBJECT_TYPE,
                    "created": created,
                    "model": SERVED_MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": clean_text},
                            "finish_reason": None,
                        }
                    ],
                }

        thread.join()

        # Final chunk
        yield {
            "id": chat_id,
            "object": CHUNK_OBJECT_TYPE,
            "created": created,
            "model": SERVED_MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


# ============================================================================
# HTTP Request Handler (OpenAI-compatible)
# ============================================================================


class OpenAIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing OpenAI v1 endpoints."""

    engine: K2HorizonEngine

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ["/v1/models", "/models"]:
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", CONTENT_TYPE_JSON)
            self.end_headers()
            response = {
                "object": "list",
                "data": [
                    {
                        "id": SERVED_MODEL_NAME,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ifm-rust-lora",
                    },
                    {
                        "id": DEFAULT_BASE_MODEL,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ifm",
                    },
                ],
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        elif self.path in ["/", "/health"]:
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", CONTENT_TYPE_JSON)
            self.end_headers()
            self.wfile.write(
                json.dumps({"status": "ok", "model": SERVED_MODEL_NAME}).encode("utf-8")
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path not in ["/v1/chat/completions", "/chat/completions"]:
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self._set_cors_headers()
            self.send_header("Content-Type", CONTENT_TYPE_JSON)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        messages = payload.get("messages", [])
        tools = payload.get("tools")
        temperature = float(payload.get("temperature", 0.2))
        top_p = float(payload.get("top_p", 0.95))
        max_tokens = int(payload.get("max_tokens", 4096))
        stream = bool(payload.get("stream", False))

        if stream:
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                for chunk in self.engine.generate_stream(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ):
                    event = f"data: {json.dumps(chunk)}\n\n"
                    self.wfile.write(event.encode("utf-8"))
                    self.wfile.flush()

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.info("Client disconnected during stream")
        else:
            resp = self.engine.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))


# ============================================================================
# CLI & Server Runner
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Serve IFM K2-Horizon-3.7B with Rust LoRA for Coding Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="IFM/K2-Horizon-3.7B",
        help="HuggingFace model ID or local directory path for base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="models/k2_horizon_rust_lora",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (e.g., 8000)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization for lower VRAM usage",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a one-shot CLI prompt test instead of starting the HTTP server",
    )

    args = parser.parse_args()

    # Initialize Engine
    engine = K2HorizonEngine(
        base_model_path=args.model_path,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
    )

    # One-shot test mode
    if args.test:
        logger.info(f"Running test prompt: '{args.test}'")
        messages = [{"role": "user", "content": args.test}]
        result = engine.generate(messages)
        print("\n" + "=" * 60)
        print("AGENT RESPONSE:")
        print("=" * 60)
        msg = result["choices"][0]["message"]
        if msg.get("reasoning_content"):
            print("[REASONING]:")
            print(msg["reasoning_content"])
            print("-" * 60)
        print(msg.get("content") or "[No text output, tool call invoked]")
        if msg.get("tool_calls"):
            print("[TOOL CALLS]:", json.dumps(msg["tool_calls"], indent=2))
        print("=" * 60)
        return

    # Start HTTP Server
    OpenAIHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    logger.info("=" * 70)
    logger.info(f"Coding Agent Server running at http://{args.host}:{args.port}")
    logger.info("Compatible with: Cline, Roo Code, Continue, Aider, Cursor, Copilot")
    logger.info("Endpoints:")
    logger.info(f"  Chat Completions: http://{args.host}:{args.port}/v1/chat/completions")
    logger.info(f"  Models List:      http://{args.host}:{args.port}/v1/models")
    logger.info("=" * 70)

    try:
        server.serve_forever()  # NOSONAR
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
