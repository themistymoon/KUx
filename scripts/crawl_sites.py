"""Fetch approved Kasetsart sites and export them as text documents."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kux.config import CrawlerConfig
from kux.crawling.site_crawler import SiteCrawler

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Kasetsart CS resources")
    parser.add_argument("urls", nargs="+", help="Seed URLs to crawl")
    parser.add_argument(
        "--output", type=str, default="data/crawled", help="Directory to store crawled text"
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CrawlerConfig()
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.max_pages is not None:
        config.max_pages = args.max_pages
    crawler = SiteCrawler(config)
    results = crawler.crawl(args.urls)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for url, text in results.items():
        filename = Path(url.replace("https://", "").replace("http://", "").replace("/", "_"))
        target = output_dir / f"{filename}.txt"
        target.write_text(text, encoding="utf-8")
    manifest = {"sources": list(results.keys())}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Crawled %s pages. Output saved to %s", len(results), output_dir)


if __name__ == "__main__":
    main()
