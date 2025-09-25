"""CLI tool to ingest documents into the FAISS vector store."""
from __future__ import annotations

import argparse
import logging

from kux.config import RAGConfig
from kux.rag.ingest import DocumentIngestor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vector store from documents")
    parser.add_argument("paths", nargs="+", help="Paths to PDF/CSV/TXT documents or directories")
    parser.add_argument("--embedding-model", type=str, default=None, help="HF embedding model")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--vector-db", type=str, default=None, help="Target directory for the FAISS index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RAGConfig()
    if args.embedding_model:
        config.embedding_model_name = args.embedding_model
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    if args.vector_db:
        config.vector_db_path = args.vector_db
    ingestor = DocumentIngestor(config)
    ingestor.ingest(args.paths)


if __name__ == "__main__":
    main()
