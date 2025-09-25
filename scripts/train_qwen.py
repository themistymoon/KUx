"""CLI entrypoint for fine-tuning Qwen using LoRA."""
from __future__ import annotations

import argparse
import json
import logging

from kux.config import TrainConfig
from kux.fine_tuning.training import SupervisedFineTuner

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-Omni with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (JSON/JSONL with chat records)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save the trained adapter"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        config = TrainConfig(**data)
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    LOGGER.info("Starting fine-tuning with config: %s", config)
    trainer = SupervisedFineTuner(config)
    trainer.prepare_datasets()
    trainer.train()


if __name__ == "__main__":
    main()
