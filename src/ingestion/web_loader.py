"""
Web page ingestion.

Uses `trafilatura` because it strips navigation/ads/boilerplate far
better than a naive BeautifulSoup text-dump — important here since
UNICEF/WHO pages are full of menus, footers and related-link widgets
that would otherwise pollute the knowledge base with noise.
"""
from __future__ import annotations

import uuid

import trafilatura

from src.schema import RawDocument


def load_web_page(
    url: str,
    organization: str,
    title: str,
    age_range: str | None = None,
    topics: list[str] | None = None,
    priority_stars: int = 3,
    license_status: str = "unknown",
) -> RawDocument | None:
    """Fetch and extract the main article text from a single URL."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        print(f"[web_loader] Failed to fetch {url}")
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )
    if not text:
        print(f"[web_loader] No extractable content at {url}")
        return None

    return RawDocument(
        doc_id=str(uuid.uuid4()),
        text=text,
        organization=organization,
        title=title,
        source_format="web",
        age_range=age_range,
        topics=topics or [],
        priority_stars=priority_stars,
        license_status=license_status,
        url=url,
    )


def load_web_manifest(manifest: list[dict]) -> list[RawDocument]:
    """
    Bulk-load every URL in a manifest, e.g.:

        [
          {
            "url": "https://www.unicef.org/egypt/parenting-hub",
            "organization": "UNICEF Egypt",
            "title": "بوابة التربية",
            "age_range": "0-18",
            "topics": ["التربية", "نمو الطفل"],
            "priority_stars": 5,
            "license_status": "red"
          },
          ...
        ]
    """
    documents = []
    for entry in manifest:
        doc = load_web_page(
            url=entry["url"],
            organization=entry["organization"],
            title=entry["title"],
            age_range=entry.get("age_range"),
            topics=entry.get("topics", []),
            priority_stars=entry.get("priority_stars", 3),
            license_status=entry.get("license_status", "unknown"),
        )
        if doc:
            documents.append(doc)
    return documents
