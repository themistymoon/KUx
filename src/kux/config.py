"""Central configuration dataclasses for KUx project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    """Configuration for supervised fine-tuning of Qwen."""

    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    dataset_path: str = "data/train.jsonl"
    output_dir: str = "outputs/finetuned-qwen"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: Optional[int] = None
    seed: int = 42
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    use_gradient_checkpointing: bool = True
    bf16: bool = True
    dataset_text_field: str = "text"


@dataclass
class RAGConfig:
    """Configuration for retrieval augmented generation."""

    vector_db_path: Path = Path("storage/vectorstore")
    chunk_size: int = 1024
    chunk_overlap: int = 80
    allowed_document_types: List[str] = field(default_factory=lambda: [".pdf", ".csv", ".txt"])
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "kasetsart_documents"
    max_retrieval_docs: int = 6


@dataclass
class CrawlerConfig:
    """Configuration for crawling approved Kasetsart CS resources."""

    allowed_domains: List[str] = field(
        default_factory=lambda: [
            "www.ku.ac.th",
            "www.cs.ku.ac.th",
            "cs.ku.ac.th",
            "registrar.ku.ac.th",
            "admission.ku.ac.th",
        ]
    )
    user_agent: str = "KUxBot/1.0 (+https://www.cs.ku.ac.th)"
    request_timeout: int = 20
    max_depth: int = 1
    max_pages: int = 20
    cache_dir: Path = Path("storage/crawler_cache")


__all__ = ["TrainConfig", "RAGConfig", "CrawlerConfig"]
