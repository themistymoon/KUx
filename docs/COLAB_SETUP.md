# Google Colab Pro+ notebook walkthrough

This guide mirrors the bundled [docs/KUx_Colab_End_to_End.ipynb](KUx_Colab_End_to_End.ipynb), which now consolidates every KUx workflow step into a single notebook. Adjust the configuration cell once, then execute the remaining sections sequentially to prepare data, rebuild the RAG store, optionally fine-tune Qwen3-Omni, and launch the multimodal chatbot without leaving Colab.

## 0. Adjust the master configuration cell

The first executable cell defines all runtime switches—repository location, crawl settings, ingestion sources, fine-tuning options, and chatbot defaults. Update it before proceeding so the subsequent cells honour your choices.

```python
CONFIG = {
    "repo_url": "https://github.com/<your-account>/KUx.git",
    "repo_dir": "/content/KUx",
    "enable_crawl": False,
    "crawl_seed_urls": [
        "https://cs.sci.ku.ac.th/",
    ],
    "crawl_max_depth": 1,
    "crawl_max_pages": 10,
    "enable_ingest": True,
    "ingest_sources": [
        "data/sample_documents",
        "data/crawled",
    ],
    "enable_finetune": False,
    "finetune_dataset": "data/train.jsonl",
    "finetune_epochs": 2,
    "default_model_key": "qwen3-omni-30b",
    "default_system_prompt": "",
    "launch_share": True,
    "launch_preload": True,
    "vector_db_dir": "storage/vectorstore",
    "adapter_dir": "outputs/finetuned-qwen",
}
```

## 1. Provision the runtime GPU

Set the notebook to **GPU ▸ A100 80 GB** (plus High-RAM if desired), then run `!nvidia-smi` to confirm Colab attached the correct accelerator.

## 2. Clone the repository (idempotent)

The clone cell reads `CONFIG["repo_url"]` and `CONFIG["repo_dir"]`, only cloning when the target directory does not exist:

```python
from pathlib import Path
from IPython import get_ipython

REPO_URL = CONFIG["repo_url"]
TARGET_DIR = Path(CONFIG["repo_dir"])

if not TARGET_DIR.exists():
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    !git clone {REPO_URL} {TARGET_DIR}

get_ipython().run_line_magic('cd', str(TARGET_DIR))
print('Working directory:', Path.cwd())
```

## 3. Install dependencies

```python
!pip install -U pip
!pip install -r requirements.txt
```

Editable installs are no longer required—the notebook defines all helpers inline.

> **Optional integrations**
>
> ```python
> # Hugging Face authentication if the Qwen model is gated
> from huggingface_hub import notebook_login
> notebook_login()
>
> # Persist artefacts to Google Drive
> from google.colab import drive
> drive.mount('/content/drive')
> ```

## 4. Configure working directories

This cell ensures the data, vector store, and adapter folders exist and writes their resolved paths back into `CONFIG` for later steps.

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
VECTOR_DB_DIR = PROJECT_ROOT / CONFIG['vector_db_dir']
ADAPTER_DIR = PROJECT_ROOT / CONFIG['adapter_dir']

for path in (DATA_DIR, VECTOR_DB_DIR, ADAPTER_DIR):
    path.mkdir(parents=True, exist_ok=True)

CONFIG['project_root'] = str(PROJECT_ROOT)
CONFIG['data_dir'] = str(DATA_DIR)
CONFIG['vector_db_dir'] = str(VECTOR_DB_DIR)
CONFIG['adapter_dir'] = str(ADAPTER_DIR)
```

A quick listing cell displays the bundled `data/sample_documents/` so you can verify the starter files.

## 5. Expand the inline KUx core logic

Before you run the crawl/ingest/fine-tune/chatbot sections, execute the notebook cell labelled **“KUx core logic within this notebook.”** It defines:

- Configuration dataclasses (`ModelOption`, `RAGConfig`, `TrainConfig`, `CrawlerConfig`).
- `SiteCrawler` for domain-restricted crawling.
- `DocumentIngestor` for building a FAISS vector store.
- `MediaInput`, `RAGPipeline`, and multimodal Qwen generator helpers.
- `SupervisedFineTuner` for LoRA training.
- `launch` for the Gradio Blocks chatbot UI.

All later cells assume these definitions are in memory.

## 6. Crawl Kasetsart CS sources (optional)

Enable crawling by setting `CONFIG["enable_crawl"] = True`. The notebook keeps the rest of the code unchanged:

```python
if CONFIG['enable_crawl']:
    crawl_seed_urls = CONFIG['crawl_seed_urls']
    if not crawl_seed_urls:
        raise ValueError('No seed URLs specified. Update CONFIG["crawl_seed_urls"].')

    crawler_config = CrawlerConfig(
        max_depth=CONFIG.get('crawl_max_depth', 1),
        max_pages=CONFIG.get('crawl_max_pages', 10),
        cache_dir=DATA_DIR / 'crawled_cache',
    )
    crawler = SiteCrawler(crawler_config)
    crawled = crawler.crawl(crawl_seed_urls)
    output_dir = DATA_DIR / 'crawled'
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (url, text) in enumerate(crawled.items(), start=1):
        target = output_dir / f'page_{idx:03d}.txt'
        target.write_text(text, encoding='utf-8')
        print(f'Saved {target.relative_to(PROJECT_ROOT)} ← {url}')
else:
    print('Skipping crawl (CONFIG["enable_crawl"] is False).')
```

## 7. Build or refresh the FAISS vector store

```python
if CONFIG['enable_ingest']:
    ingest_config = RAGConfig(vector_db_path=VECTOR_DB_DIR)
    ingestor = DocumentIngestor(ingest_config)
    sources = [str((PROJECT_ROOT / path)) for path in CONFIG['ingest_sources']]
    vector_store = ingestor.ingest(sources)
    print('Vector store ready:', VECTOR_DB_DIR)
else:
    print('Skipping ingestion (CONFIG["enable_ingest"] is False).')
```

## 8. Optional: fine-tune Qwen with LoRA adapters

```python
if CONFIG['enable_finetune']:
    train_config = TrainConfig(
        dataset_path=CONFIG['finetune_dataset'],
        num_train_epochs=CONFIG.get('finetune_epochs', 2),
        output_dir=str(ADAPTER_DIR),
    )
    finetuner = SupervisedFineTuner(train_config)
    finetuner.prepare_datasets()
    finetuner.train()
else:
    print('Skipping fine-tuning (CONFIG["enable_finetune"] is False).')
```

## 9. Launch the multimodal KUx chatbot

The final section preloads the requested model, then launches Gradio with the inline `launch` helper.

```python
launch(
    vector_db_path=str(VECTOR_DB_DIR),
    adapter_dir=str(ADAPTER_DIR),
    default_model_key=CONFIG.get('default_model_key', 'qwen3-omni-30b'),
    default_system_prompt=CONFIG.get('default_system_prompt', ''),
    share=CONFIG.get('launch_share', True),
    preload_default=CONFIG.get('launch_preload', True),
)
```

Gradio prints both the local and public URLs directly in the output cell—open the share link in a new browser tab to interact with KUx while leaving the launch cell running.
