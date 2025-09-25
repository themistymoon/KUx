# KUx – Kasetsart CS Multimodal Assistant

KUx is a retrieval-augmented, Kasetsart University–focused assistant built on top of **Qwen3-Omni-30B-A3B-Instruct**. The project provides:

- A LoRA fine-tuning pipeline tailored for custom Kasetsart Computer Science supervision data.
- Retrieval augmented generation (RAG) over small PDFs (≤ 20 pages), CSVs, and crawler harvested content from approved Kasetsart domains.
- Utilities to crawl and cache official Kasetsart websites for up-to-date information.
- A Gradio demo that emulates a chatbot-style experience suitable for deployment on Google Colab Pro+ (A100 80GB).

> **Mission reminder:** KUx must be a trustworthy all-round helper for Kasetsart students, while prioritising correctness for the Computer Science programme. Retrieval sources therefore default to official KU domains and local curated datasets.

## Repository layout

```
├── configs/               # Place optional YAML/JSON configuration overrides here
├── data/                  # (ignored) Training data, crawled text, and ingested docs
├── docs/                  # How-to guides
├── scripts/               # Command line entrypoints for training, RAG and web UI
├── src/kux/               # Python package with training, RAG, crawling and UI modules
└── requirements.txt       # Runtime and training dependencies
```

## Quick start on Google Colab Pro+

1. **Provision the runtime** – open a fresh Colab notebook, switch the hardware accelerator to **GPU (A100 80GB)**, and optionally enable High-RAM.
2. **Clone and install:**
   ```bash
   !git clone https://github.com/themistymoon/KUx.git 
   %cd KUx
   !pip install -U pip
   !pip install -r requirements.txt
   ```
3. **(Optional) Mount Google Drive** to keep the FAISS index and LoRA adapters between sessions:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Authenticate with Hugging Face** if your account needs gated access:
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```
5. **Upload your supervision dataset** to `data/train.jsonl` using either the chat-format (`messages`) or instruction-format (`instruction`/`response`).
6. **(Optional) Crawl official KU CS resources** for fresh retrieval data:
   ```bash
   !python scripts/crawl_sites.py https://www.cs.ku.ac.th --output data/crawled --max-depth 1 --max-pages 10
   ```
7. **Ingest PDFs, CSVs, and crawled text into FAISS:**
   ```bash
   !python scripts/build_vector_store.py data/crawled path/to/pdfs path/to/csvs --vector-db storage/vectorstore
   ```
8. **Fine-tune Qwen with LoRA adapters:**
   ```bash
   !python scripts/train_qwen.py --dataset data/train.jsonl --output-dir outputs/finetuned-qwen
   ```
9. **Launch the Gradio chatbot** (public URL printed in the notebook):
   ```bash
   !python scripts/run_chatbot.py --vector-db storage/vectorstore --adapter outputs/finetuned-qwen
   ```

For a detailed, copy-paste-ready notebook walkthrough (including advanced overrides and export tips) see [docs/COLAB_SETUP.md](docs/COLAB_SETUP.md).

## Fine-tuning pipeline

`src/kux/fine_tuning/training.py` implements a LoRA fine-tuning workflow using 4-bit quantisation for memory efficiency on a single A100 80GB GPU. Key features include:

- Automatic formatting of chat-style or instruction-following datasets into the Qwen chat template.
- Configurable LoRA rank, dropout, sequence length and logging cadence via `TrainConfig`.
- Gradient checkpointing and BF16 training for large sequence support.
- Persisted adapter weights and tokenizer saved under `outputs/finetuned-qwen/` by default.

Adjust hyperparameters by editing `TrainConfig` (see `src/kux/config.py`) or by supplying a JSON config to `scripts/train_qwen.py`.

## Retrieval augmented generation (RAG)

- **Document ingestion:** `DocumentIngestor` (`src/kux/rag/ingest.py`) accepts directories or file paths for PDFs (≤ 20 pages recommended), CSVs, and text files. Documents are chunked with a recursive splitter and embedded with `sentence-transformers/all-MiniLM-L6-v2` by default.
- **Vector store:** Documents are embedded into a FAISS index stored under `storage/vectorstore/` (configurable). The same index is reused by the chatbot and any downstream tools.
- **Question answering:** `RAGPipeline` (`src/kux/rag/pipeline.py`) retrieves top-k passages and builds Qwen-aligned prompts to guarantee factual grounding. If no supporting passages are found the assistant explicitly states its uncertainty.

## Focused crawler for KU sources

`scripts/crawl_sites.py` uses `SiteCrawler` (`src/kux/crawling/site_crawler.py`) to fetch, cache, and clean HTML from approved Kasetsart domains only. Update `CrawlerConfig.allowed_domains` (in `src/kux/config.py`) to whitelist additional official sources. Crawled pages are exported as UTF-8 `.txt` files ready for ingestion.

## Chatbot demo

The demo (`src/kux/chatbot/app.py`) exposes a Gradio chat interface styled as a KUx-branded assistant. It lazily initialises the RAG pipeline and streams answers grounded in the FAISS index. When running inside Colab the `launch()` helper automatically shares a public URL.

## Deployment tips

- **Persistent storage:** Mount Google Drive in Colab to persist adapters (`outputs/`) and vector stores (`storage/`).
- **Serving outside Colab:** Run `python scripts/run_chatbot.py --share` via Gradio sharing or host behind a reverse proxy (e.g., Cloud Run) after exporting the trained adapter and FAISS index.
- **Updating knowledge:** Re-run the crawler and ingestion scripts whenever new syllabi or announcements are published; the chatbot will use the refreshed vector store without retraining.

## License

This project is released under the [MIT License](LICENSE).
