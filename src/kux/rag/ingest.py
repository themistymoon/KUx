"""Document ingestion utilities for KUx retrieval augmented generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import RAGConfig

LOGGER = logging.getLogger(__name__)


class DocumentIngestor:
    """Ingest PDFs, CSVs and text files into a FAISS vector store."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.vector_db_path = Path(self.config.vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def _resolve_loader(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        if suffix == ".csv":
            return CSVLoader(str(file_path))
        if suffix in {".txt", ".md"}:
            return TextLoader(str(file_path))
        raise ValueError(f"Unsupported file type for ingestion: {suffix}")

    def _load_documents(self, path: Path):
        loader = self._resolve_loader(path)
        documents = loader.load()
        LOGGER.info("Loaded %s documents from %s", len(documents), path)
        return documents

    def ingest(self, sources: Iterable[str]) -> FAISS:
        """Ingest the provided sources into the FAISS vector store."""

        docs = []
        for source in sources:
            path = Path(source)
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.suffix.lower() in self.config.allowed_document_types:
                        docs.extend(self._load_documents(child))
            elif path.suffix.lower() in self.config.allowed_document_types:
                docs.extend(self._load_documents(path))
            else:
                LOGGER.warning("Skipping unsupported file: %s", path)
        if not docs:
            raise RuntimeError("No documents were ingested. Check your source paths.")

        LOGGER.info("Splitting %s documents into chunks", len(docs))
        chunks = self.splitter.split_documents(docs)
        LOGGER.info("Creating vector store with %s chunks", len(chunks))

        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        vector_store.save_local(str(self.vector_db_path))
        LOGGER.info("Vector store saved to %s", self.vector_db_path)
        return vector_store


__all__ = ["DocumentIngestor"]
