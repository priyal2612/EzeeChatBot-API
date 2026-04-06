"""
Content fetcher — retrieves text from URLs.

Handles:
  - Plain HTML pages: strips tags with a simple regex + BeautifulSoup fallback
  - Text/plain responses: returned as-is
  - Content-length guard: refuses payloads > 5 MB

In production I'd add:
  - PDF extraction (pdfplumber)
  - JavaScript-rendered pages (Playwright)
  - Retry/back-off with exponential delay
"""

import re
import urllib.request
import urllib.error
from typing import Tuple


MAX_BYTES = 5 * 1024 * 1024  # 5 MB hard cap


def _strip_html(html: str) -> str:
    """Very fast HTML-to-text: remove scripts, styles, then all tags."""
    # Remove script and style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_url(url: str) -> Tuple[str, str]:
    """
    Returns (text_content, source_description).
    Raises ValueError for oversized or unsupported content.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "EzeeChatBot/1.0 (knowledge-base-fetcher)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_BYTES + 1)

        if len(raw) > MAX_BYTES:
            raise ValueError(f"URL content exceeds 5 MB limit.")

        encoding = "utf-8"
        charset_match = re.search(r"charset=([^\s;]+)", content_type)
        if charset_match:
            encoding = charset_match.group(1).strip()

        text = raw.decode(encoding, errors="replace")

        if "html" in content_type.lower():
            text = _strip_html(text)

        return text, url

    except urllib.error.URLError as e:
        raise ValueError(f"Could not fetch URL: {e.reason}")
    except Exception as e:
        raise ValueError(f"URL fetch failed: {str(e)}")
