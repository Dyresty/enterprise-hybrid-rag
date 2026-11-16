"""Minimal OpenAI-compatible chat server for a local Hugging Face model.

Lets the RAG pipeline and the benchmark talk to the base or the fine-tuned model
through the same LLM_BASE_URL setting used for OpenAI, vLLM or Ollama.

    python -m finetune.serve --model finetune/out/merged --name cordis-qwen --port 8100
    LLM_BASE_URL=http://127.0.0.1:8100/v1 LLM_MODEL=cordis-qwen python -m app.evaluation.benchmark --generation
"""
import argparse
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 512


def create_app(model_path, name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0}).eval()
    lock = threading.Lock()  # one generation at a time on a single GPU
    app = FastAPI(title=f"Local LLM server ({name})")

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        inputs = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        sampling = {"do_sample": True, "temperature": request.temperature} if request.temperature > 0 else {"do_sample": False}
        with lock, torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=request.max_tokens, **sampling)

        prompt_tokens = inputs["input_ids"].shape[1]
        text = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output.shape[1] - prompt_tokens,
                "total_tokens": output.shape[1],
            },
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.name), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
