"""Launch the KUx Gradio chatbot."""
from __future__ import annotations

import argparse
from typing import Optional

from kux.chatbot.app import launch
from kux.config import MODEL_OPTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the KUx chatbot demo")
    parser.add_argument(
        "--vector-db",
        type=str,
        default=None,
        help="Path to the FAISS vector store directory (defaults to storage/vectorstore)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Directory containing LoRA adapters produced by scripts/train_qwen.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[option.key for option in MODEL_OPTIONS],
        default=None,
        help="Default base model selection (matches ModelOption.key)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the default system prompt shown in the UI",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing (useful on Colab)",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Hostname/interface to bind the Gradio server to",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port for the Gradio server",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()
    launch(
        vector_db_path=parsed.vector_db,
        adapter_dir=parsed.adapter,
        default_model_key=parsed.model,
        default_system_prompt=parsed.system_prompt,
        share=parsed.share,
        server_name=parsed.server_name,
        server_port=parsed.server_port,
    )


if __name__ == "__main__":
    main()
