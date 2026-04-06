from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN_DEFAULT_MAX_NEW_TOKENS", "192"))
DEFAULT_TEMPERATURE = float(os.getenv("QWEN_DEFAULT_TEMPERATURE", "0.2"))
TORCH_THREADS = int(os.getenv("QWEN_TORCH_THREADS", "2"))
DEVICE = os.getenv("QWEN_DEVICE", "auto")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    model_id: str
    text: str


class GenerateBatchRequest(BaseModel):
    requests: list[GenerateRequest] = Field(min_length=1, max_length=32)


class GenerateBatchResult(BaseModel):
    text: str


class GenerateBatchResponse(BaseModel):
    model_id: str
    results: list[GenerateBatchResult]


def resolve_device() -> str:
    requested = DEVICE.strip().lower()
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_chat_prompt(tokenizer, prompt: str, system_prompt: str | None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    return prompt


def load_model() -> tuple[Any, Any]:
    torch.set_num_threads(TORCH_THREADS)
    resolved_device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_config_path = Path(MODEL_ID) / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, MODEL_ID)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model, resolved_device = load_model()
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.model_lock = threading.Lock()
    app.state.resolved_device = resolved_device
    yield


app = FastAPI(title="LLMOps Local Qwen Inference", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "requested_device": DEVICE,
        "resolved_device": str(app.state.resolved_device),
    }


def run_generation_batch(tokenizer, model, requests: list[GenerateRequest]) -> list[str]:
    resolved_device = resolve_device()
    prompt_texts = [build_chat_prompt(tokenizer, item.prompt, item.system_prompt) for item in requests]
    encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    encoded = {key: value.to(resolved_device) for key, value in encoded.items()}

    max_new_tokens = max(item.max_new_tokens or DEFAULT_MAX_NEW_TOKENS for item in requests)
    temperatures = [item.temperature if item.temperature is not None else DEFAULT_TEMPERATURE for item in requests]
    unique_temperatures = {round(float(temperature), 6) for temperature in temperatures}
    if len(unique_temperatures) != 1:
        raise ValueError("All requests in a batch must share the same temperature")

    temperature = temperatures[0]
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[-1]
    return [
        tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True).strip()
        for sequence in generated
    ]


def run_generation_requests(tokenizer, model, requests: list[GenerateRequest], model_lock) -> list[str]:
    grouped_requests: dict[tuple[str | None, int, float], list[tuple[int, GenerateRequest]]] = {}
    for index, request in enumerate(requests):
        group_key = (
            request.system_prompt,
            request.max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
            float(request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE),
        )
        grouped_requests.setdefault(group_key, []).append((index, request))

    responses = [""] * len(requests)
    with model_lock:
        for group in grouped_requests.values():
            _, group_requests = zip(*group)
            texts = run_generation_batch(tokenizer, model, list(group_requests))
            for (index, _), text in zip(group, texts):
                responses[index] = text
    return responses


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    tokenizer = app.state.tokenizer
    model = app.state.model
    model_lock = app.state.model_lock

    text = run_generation_requests(tokenizer, model, [payload], model_lock)[0]
    return GenerateResponse(model_id=MODEL_ID, text=text)


@app.post("/v1/generate_batch", response_model=GenerateBatchResponse)
def generate_batch(payload: GenerateBatchRequest) -> GenerateBatchResponse:
    tokenizer = app.state.tokenizer
    model = app.state.model
    model_lock = app.state.model_lock

    texts = run_generation_requests(tokenizer, model, payload.requests, model_lock)
    return GenerateBatchResponse(
        model_id=MODEL_ID,
        results=[GenerateBatchResult(text=text) for text in texts],
    )
