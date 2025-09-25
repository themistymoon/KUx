"""Simple focused crawler for Kasetsart University domains."""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ..config import CrawlerConfig

LOGGER = logging.getLogger(__name__)


class SiteCrawler:
    """Breadth-first crawler constrained to approved Kasetsart domains."""

    def __init__(self, config: CrawlerConfig | None = None) -> None:
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetching helpers
    # ------------------------------------------------------------------
    def _is_allowed(self, url: str) -> bool:
        domain = urlparse(url).netloc
        return domain in self.config.allowed_domains

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def fetch(self, url: str) -> str:
        if not self._is_allowed(url):
            raise ValueError(f"URL domain not allowed: {url}")
        cache_path = self._cache_path(url)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore")
        response = self.session.get(url, timeout=self.config.request_timeout)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join(chunk.strip() for chunk in soup.stripped_strings)
        return text

    def extract_links(self, base_url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            absolute = urljoin(base_url, href)
            if self._is_allowed(absolute):
                links.append(absolute)
        return links

    # ------------------------------------------------------------------
    # Crawling orchestration
    # ------------------------------------------------------------------
    def crawl(self, seeds: Iterable[str]) -> Dict[str, str]:
        """Crawl starting from the seed URLs and return url->text mapping."""

        visited: Set[str] = set()
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        results: Dict[str, str] = {}
        while queue and len(results) < self.config.max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > self.config.max_depth:
                continue
            try:
                html = self.fetch(url)
                text = self.extract_text(html)
            except Exception as exc:  # pragma: no cover - network issues
                LOGGER.warning("Failed to crawl %s: %s", url, exc)
                continue
            visited.add(url)
            results[url] = text
            if depth < self.config.max_depth:
                for link in self.extract_links(url, html):
                    if link not in visited:
                        queue.append((link, depth + 1))
        return results


__all__ = ["SiteCrawler"]
