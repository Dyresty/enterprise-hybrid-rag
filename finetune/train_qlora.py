"""QLoRA fine-tuning of a small instruct model for grounded, cited RAG answers.

The base model is loaded in 4-bit NF4 and frozen; only low-rank adapters (LoRA) on the
attention and MLP projections are trained. Loss is computed on the assistant answer only,
not on the long retrieved context in the prompt.

    python -m finetune.train_qlora
    python -m finetune.train_qlora --merge-only   # merge saved adapter into the base model
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "finetune/out/adapter"
MERGED_DIR = "finetune/out/merged"


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    data = load_dataset(
        "json",
        data_files={"train": "finetune/data/train.jsonl", "validation": "finetune/data/val.jsonl"},
    ).remove_columns(["project_id"])

    # Long examples push a 12 GB GPU into shared system memory (10x slower), and truncation
    # would cut the answer, which is at the end. So over-long examples are dropped instead.
    def fits(example):
        ids = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True, return_dict=True)
        return len(ids["input_ids"]) <= args.max_length

    before = len(data["train"])
    data = data.filter(fits)
    print(f"Training examples: {len(data['train'])} of {before} (<= {args.max_length} tokens)")

    config = SFTConfig(
        output_dir="finetune/out/checkpoints",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # long sequences x 152k vocab: logits dominate memory
        gradient_accumulation_steps=8,  # effective batch size 8
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=True,  # prompt/completion format: train on the answer only
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    with open("finetune/out/train_log.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)


def merge():
    """Merge the adapter into full-precision base weights so Ollama can import one model."""
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="short run to measure speed/memory")
    args = parser.parse_args()

    if not args.merge_only:
        train(args)
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        torch.cuda.empty_cache()
        if args.max_steps > 0:
            return
    merge()


if __name__ == "__main__":
    main()
