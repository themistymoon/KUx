# Google Colab Pro+ deployment guide

This walkthrough shows how to bootstrap KUx on a **Google Colab Pro+** runtime with an **A100 80GB** GPU. The same steps also work on other high-memory accelerators, but the LoRA recipe and quantisation defaults have been tuned for the Colab Pro+ offering.

## 0. Start from a clean notebook

1. Create a new Colab notebook → **Runtime ▸ Change runtime type** → set **Hardware accelerator** to **GPU** (A100 80GB).
2. (Optional) Enable **High-RAM** runtimes if available to comfortably cache the vector store and crawler output.

## 1. Environment preparation

```python
# Inspect the GPU to confirm you received an A100 80GB device
!nvidia-smi

# Pull the project and install dependencies
!git clone https://github.com/<your-account>/KUx.git
%cd KUx
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
```
> The editable install registers the `kux` package on `PYTHONPATH`, allowing `python scripts/*.py` commands to import the project modules without extra setup.

> **Persisting artefacts:** Mount Google Drive early in the notebook to keep trained adapters (`outputs/`) and FAISS indexes (`storage/`) between sessions:
> `from google.colab import drive; drive.mount('/content/drive')`

> **Transformer nightly (for multimodal I/O):** Qwen3-Omni’s audio/video tooling lives on the latest Transformers main branch.
> After installing the base requirements, run `pip install -U "transformers@git+https://github.com/huggingface/transformers"`
> to ensure the multimodal loaders are available.

If you rely on gated Hugging Face models, authenticate once per session:

```python
from huggingface_hub import notebook_login
notebook_login()  # prompts for your HF token
```

## 2. Prepare training data

Create or upload a supervision file at `data/train.jsonl`. Each line must be a JSON object in one of the supported formats:

```json
{"messages": [
  {"role": "system", "content": "You are KUx, an assistant for Kasetsart CS students."},
  {"role": "user", "content": "<student question>"},
  {"role": "assistant", "content": "<grounded answer>"}
]}
```

or

```json
{"instruction": "<prompt>", "input": "<optional context>", "response": "<answer>"}
```

Keep prompts and answers anchored to **Kasetsart University Computer Science** knowledge so that fine-tuning reinforces the programme’s guidelines.

## 3. Collect official references (optional but recommended)

Fetch up-to-date announcements, course descriptions, or curriculum details from approved Kasetsart domains:

```bash
!python scripts/crawl_sites.py https://www.cs.ku.ac.th \
    --output data/crawled \
    --max-depth 1 \
    --max-pages 10
```

Outputs:

- Raw HTML cache → `storage/crawler_cache/`
- Cleaned plain text → `data/crawled/`

Feel free to run the crawler on other whitelisted URLs by editing `src/kux/config.py` (`CrawlerConfig.allowed_domains`).

## 4. Ingest PDFs, CSVs, and crawled text

Populate the FAISS vector store that powers the RAG pipeline. You can combine multiple sources in one command:

```bash
!python scripts/build_vector_store.py \
    data/crawled \
    /content/drive/MyDrive/kux_docs/pdfs \
    /content/drive/MyDrive/kux_docs/csvs \
    --vector-db storage/vectorstore
```

The script recursively scans each path, filters for `.pdf`, `.csv`, `.txt`, and `.md` files, chunks them, and writes the embeddings into `storage/vectorstore/`.

Re-run the ingestion command whenever you add or modify documents. The existing index will be updated in-place.

## 5. Fine-tune Qwen with LoRA adapters

Launch training once the dataset is ready. The defaults in `TrainConfig` are sized for a single A100 80GB GPU, but you can override them through CLI flags:

```bash
!python scripts/train_qwen.py \
    --dataset data/train.jsonl \
    --output-dir outputs/finetuned-qwen \
    --num-epochs 3 \
    --learning-rate 2e-4
```

Training artefacts saved to `outputs/finetuned-qwen/` include:

- `adapter_config.json` and `adapter_model.safetensors` (LoRA weights)
- `tokenizer.json` plus tokenizer configs
- `training_args.bin` for reproducibility

Mount Google Drive or download this folder to persist the adapters for later use.

## 6. Launch the Gradio chatbot demo

With the vector store and LoRA adapter available, start the Colab-hosted interface:

```bash
!python scripts/run_chatbot.py \
    --vector-db storage/vectorstore \
    --adapter outputs/finetuned-qwen \
    --model qwen3-omni-30b \
    --share
```

`--vector-db`, `--adapter`, and `--model` are optional. If you omit them, KUx will still load the base multimodal checkpoint
and answer questions, but responses will not be grounded in Kasetsart documents until the FAISS store and adapters are provided.

Gradio prints both a local URL and a **public share URL**. Open the public link to chat with KUx. The interface now surfaces
uploaders for images, audio, and video—leave the text box empty if you want KUx to perform OCR, object grounding, speech
recognition/translation, audio captioning, music analysis, or full audio-visual dialogue/function-call reasoning on the
attachments. Each response continues to cite retrieved chunks so you can validate accuracy.

If you restart the runtime, rerun sections 1, 4, 5, and 6 (mount Drive first to reuse stored artefacts).

## 7. (Optional) Export for other deployments

- **FAISS index:** `storage/vectorstore/` → copy to your server or object storage.
- **Adapters:** `outputs/finetuned-qwen/` → load with `peft.PeftModel.from_pretrained` alongside the base Qwen model.
- **Environment file:** `requirements.txt` ensures reproducible installs outside Colab.

Once exported, you can launch `scripts/run_chatbot.py` on a workstation or cloud VM with the same arguments and bypass Colab entirely.
