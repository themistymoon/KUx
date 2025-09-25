"""Gradio chat interface for the KUx assistant."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Tuple

import gradio as gr

from ..config import RAGConfig, TrainConfig
from ..rag.pipeline import RAGPipeline

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_pipeline() -> RAGPipeline:
    LOGGER.info("Initialising RAG pipeline for KUx chatbot")
    return RAGPipeline(RAGConfig(), TrainConfig())


def respond(message: str, history: List[Tuple[str, str]]) -> str:
    pipeline = load_pipeline()
    answer = pipeline.answer(message)
    return answer


def launch() -> None:
    description = (
        "KUx is a retrieval-augmented assistant for Kasetsart University Computer Science students."
    )
    chat = gr.ChatInterface(
        fn=respond,
        title="KUx – Kasetsart CS Assistant",
        description=description,
        theme=gr.themes.Default(primary_hue=gr.themes.colors.green),
        retry_btn=None,
    )
    chat.launch(server_name="0.0.0.0", server_port=7860)


__all__ = ["launch"]
