"""Retrieval augmented generation pipeline for KUx."""
from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Optional

import torch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import RAGConfig, TrainConfig

LOGGER = logging.getLogger(__name__)


class LocalQwenGenerator:
    """Wrapper around a local Qwen causal LM for inference."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        LOGGER.info("Loading Qwen generator from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.pipe(prompt)
        text = outputs[0]["generated_text"]
        return text[len(prompt) :].strip()


class RAGPipeline:
    """High level helper for running retrieval augmented generation."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        train_config: Optional[TrainConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.rag_config = rag_config or RAGConfig()
        self.train_config = train_config or TrainConfig()
        self.system_prompt = system_prompt or dedent(
            """
            You are KUx, an omniscient assistant for Kasetsart University Computer Science students.
            Answer with verified facts from Kasetsart University sources. If unsure, state that you do not know.
            """
        ).strip()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.rag_config.embedding_model_name)
        self.vector_store = self._load_vector_store(self.rag_config.vector_db_path)
        self.generator = LocalQwenGenerator(self.train_config.output_dir)

    def _load_vector_store(self, path: Path | str) -> FAISS:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {path}. Run the ingestion pipeline first."
            )
        return FAISS.load_local(
            str(path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _format_context(self, documents: Iterable[Document]) -> str:
        context_blocks: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            block = dedent(
                f"""
                [Document {idx} | Source: {source}]
                {doc.page_content.strip()}
                """
            ).strip()
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def build_prompt(self, question: str, documents: List[Document]) -> str:
        context = self._format_context(documents)
        prompt = dedent(
            f"""
            <|im_start|>system
            {self.system_prompt}
            <|im_end|>
            <|im_start|>user
            Question: {question}

            Use the following context to ground your answer:
            {context}
            <|im_end|>
            <|im_start|>assistant
            """
        )
        return prompt

    def answer(self, question: str, top_k: Optional[int] = None) -> str:
        top_k = top_k or self.rag_config.max_retrieval_docs
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        documents = retriever.get_relevant_documents(question)
        if not documents:
            return "I could not find supporting documents for that question."
        prompt = self.build_prompt(question, documents)
        answer = self.generator.generate(prompt)
        return answer


__all__ = ["RAGPipeline", "LocalQwenGenerator"]
