# KUx – Kasetsart CS Multimodal Assistant

KUx is a retrieval-augmented, Kasetsart University–focused assistant powered by **Qwen3-Omni-30B-A3B-Instruct**. The project now ships as a single Google Colab notebook that installs dependencies, crawls Kasetsart sources, ingests PDFs/CSVs into FAISS, optionally fine-tunes Qwen with LoRA, and launches a multimodal ChatGPT-style demo that runs comfortably on **Google Colab Pro+ (A100 80 GB)**.

> **Mission reminder:** KUx must be a trustworthy all-round helper for Kasetsart students while guaranteeing correctness for the Computer Science programme. Retrieval sources therefore default to official KU domains and curated local datasets.

## Repository layout

```
├── data/                    # Sample PDFs/CSVs plus a workspace for your own corpora
├── docs/                    # Colab notebook & detailed walkthrough
├── requirements.txt         # Runtime/training dependencies for the notebook
└── .gitignore               # Keeps large artefacts (vector stores, adapters) out of git
```

Every piece of project logic lives inside [`docs/KUx_Colab_End_to_End.ipynb`](docs/KUx_Colab_End_to_End.ipynb); you no longer need to install a Python package or import helper modules.

## Quick start on Google Colab Pro+

1. **Open the notebook** – upload or clone the repository in Colab and open [`docs/KUx_Colab_End_to_End.ipynb`](docs/KUx_Colab_End_to_End.ipynb). All steps run from this single file.
2. **Edit the master `CONFIG` cell** – choose crawl/ingestion/fine-tuning behaviour, adjust source directories, and override chatbot defaults before executing the rest of the notebook.
3. **Provision the runtime** – set the Colab hardware accelerator to **GPU (A100 80 GB)** and enable High-RAM. The first code cell prints `nvidia-smi` so you can confirm the runtime.
4. **Install dependencies** – run `!pip install -U pip` followed by `!pip install -r requirements.txt`. No editable install is required because the notebook defines every class and function inline.
5. **Configure storage paths** – the setup cells create `data/`, `storage/vectorstore/`, and `outputs/` folders inside the repository so crawled pages, FAISS indexes, and LoRA adapters persist for the session.
6. **Review the inline KUx core logic** – a dedicated section expands the crawler, RAG ingestion utilities, multimodal Qwen pipeline, fine-tuning helper, and Gradio app directly inside the notebook for transparency.
7. **Populate knowledge sources** – toggle the crawler via `CONFIG["enable_crawl"]` to pull fresh pages from approved KU domains, and drop your own PDFs, CSVs, or Markdown notes inside `data/` (starter files live in `data/sample_documents/`).
8. **Build the vector store** – execute the ingestion cell (enabled by default). It initialises the inline `DocumentIngestor`, walks each configured source directory, embeds the chunks, and saves the FAISS index to `storage/vectorstore/`.
9. **Fine-tune Qwen (optional)** – enable `CONFIG["enable_finetune"]` if you have a chat-style dataset (`data/train.jsonl`). The notebook’s `SupervisedFineTuner` cell loads Qwen3-Omni with LoRA and writes adapters to `outputs/`.
10. **Launch the chatbot** – run the final Serve KUx cell. The inline `launch` helper preloads the requested model (Qwen3-Omni or the text-only fallback) and brings up a Gradio Blocks UI with multimodal uploaders. Gradio prints both the local and public `--share` URLs directly in the output cell so you can open the ChatGPT-style interface.

Each step streams its logs/output inline—no external terminals or scripts required.

## Multimodal chatbot controls

The Gradio interface exposed by the notebook includes:

- **Model selector** – choose between the multimodal `Qwen3-Omni-30B-A3B-Instruct` and the text-only `gpt-oss-120b`. The selected option determines whether image/audio/video uploaders are active for the next turn.
- **Fine-tune toggle** – enable or disable LoRA adapters saved in `outputs/`. When disabled, the base model answers using only the retrieved context.
- **System prompt editor** – customise the assistant’s default instructions (pre-populated with the Kasetsart CS mission prompt).
- **Media uploaders** – attach images (OCR, object grounding, image math), audio clips (speech recognition, translation, captioning, sound/music analysis), or videos (audio-visual QA/interactions). Leave the text box empty to request pure media analysis.

The backend keeps Retrieval-Augmented Generation active so that every answer cites supporting passages from the FAISS index. When the vector store is missing, KUx will still respond but clearly note that no grounding evidence was found.

## Sample data bundle

`data/sample_documents/` contains a CSV and text digest of Kasetsart Computer Science highlights. Use them to validate the ingestion flow before adding your own PDFs, syllabi, or scraped pages. The notebook automatically picks up anything placed under `data/` if the extension is supported (`.pdf`, `.csv`, `.txt`, `.md`).

## Want to inspect or tweak the logic?

Search for the **“KUx core logic within this notebook”** heading. The following code cells expose the dataclasses, crawler, ingestion utilities, retrieval pipeline, fine-tuning helper, and Gradio app in pure Python so you can modify behaviour directly in Colab without juggling separate modules.

## License

This project is released under the [MIT License](LICENSE).
