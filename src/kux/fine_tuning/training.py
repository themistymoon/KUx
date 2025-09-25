"""Supervised fine-tuning pipeline for Qwen3-Omni-30B."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..config import TrainConfig

try:  # Optional imports used only during training
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover - only raised when deps missing
    raise ImportError(
        "peft is required for LoRA training. Install with `pip install peft`."
    ) from exc


ChatMessages = List[Dict[str, str]]


def _normalise_sample(example: Dict[str, Any]) -> str:
    """Normalise a dataset row into a chat-style plain text sample."""

    if "messages" in example:
        messages: ChatMessages = example["messages"]
        return messages_to_text(messages)

    if {"instruction", "response"}.issubset(example):
        system_prompt = example.get("system", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        if example.get("input"):
            messages.insert(2, {"role": "user", "content": example["input"]})
        return messages_to_text(messages)

    text_field = example.get("text")
    if text_field:
        return str(text_field)

    raise ValueError(
        "Unsupported dataset schema. Expected `messages`, `text` or `instruction`/`response` columns."
    )


def messages_to_text(messages: ChatMessages) -> str:
    """Convert chat messages into model-ready text using the tokenizer template."""

    formatted: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    formatted.append("<|im_start|>assistant\n")
    return "\n".join(formatted)


class SupervisedFineTuner:
    """LoRA supervised fine-tuning helper for the Qwen 3 Omni family."""

    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()
        set_seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: Optional[torch.nn.Module] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    # ------------------------------------------------------------------
    # Dataset utilities
    # ------------------------------------------------------------------
    def prepare_datasets(
        self, eval_split: Optional[float] = 0.05, streaming: bool = False
    ) -> None:
        """Load datasets, normalise formatting and tokenize them."""

        dataset_path = self.config.dataset_path
        data_files: Dict[str, str]
        path = Path(dataset_path)
        if path.is_dir():
            data_files = {"train": str(path / "train.jsonl")}
        else:
            data_files = {"train": str(path)}

        if streaming:
            dataset = load_dataset("json", data_files=data_files, streaming=True)[
                "train"
            ]
            raise NotImplementedError("Streaming datasets are not supported in this release.")

        dataset_dict = load_dataset("json", data_files=data_files)
        field_name = self.config.dataset_text_field
        train_ds = dataset_dict["train"].map(
            lambda example: {field_name: _normalise_sample(example)}
        )
        if eval_split:
            split = train_ds.train_test_split(test_size=eval_split, seed=self.config.seed)
            self.train_dataset = self._tokenize(split["train"])
            self.eval_dataset = self._tokenize(split["test"])
        else:
            self.train_dataset = self._tokenize(train_ds)
            self.eval_dataset = None

    def _tokenize(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset into model inputs."""

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                examples[self.config.dataset_text_field],
                truncation=True,
                max_length=self.config.max_seq_length,
            )

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[self.config.dataset_text_field],
        )

    # ------------------------------------------------------------------
    # Model utilities
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        torch_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        quantization_config: Dict[str, Any] = {}
        if self.config.load_in_4bit:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch_dtype,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch_dtype,
            **quantization_config,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------
    def train(self) -> None:
        if self.train_dataset is None:
            self.prepare_datasets()
        if self.model is None:
            self._load_model()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            bf16=self.config.bf16,
            evaluation_strategy="steps" if self.eval_dataset is not None else "no",
            eval_steps=self.config.eval_steps,
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        with open(Path(self.config.output_dir) / "train_config.json", "w", encoding="utf-8") as fp:
            json.dump(asdict(self.config), fp, indent=2)


__all__ = ["SupervisedFineTuner", "TrainConfig"]
