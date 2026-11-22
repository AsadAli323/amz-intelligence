"""Understand what a visitor typed: a product link, an ASIN, or a plain product name."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

_ASIN_IN_URL = re.compile(r"(?:/dp/|/gp/product/|/gp/aw/d/|/product/|/d/)([A-Z0-9]{10})(?:[/?#]|$)", re.I)
_BARE_ASIN = re.compile(r"^[A-Z0-9]{10}$")


@dataclass
class Resolved:
    kind: str                 # "link", "asin" or "name"
    asin: str | None = None
    title_hint: str | None = None


def _slug_title(path: str) -> str | None:
    """Amazon links look like /Brand-Product-Name/dp/B0XXXXXXXX: the first segment is the title."""
    parts = [p for p in path.split("/") if p]
    if "dp" in parts and parts.index("dp") > 0:
        slug = unquote(parts[parts.index("dp") - 1])
        title = re.sub(r"[-_+]+", " ", slug).strip()
        return title if len(title) > 3 else None
    return None


def parse_input(text: str) -> Resolved:
    t = (text or "").strip()
    if not t:
        return Resolved("name")
    if re.match(r"^(https?://|www\.)", t, re.I) or "amazon." in t.lower() or "amzn." in t.lower():
        url = t if "://" in t else "https://" + t
        parsed = urlparse(url)
        m = _ASIN_IN_URL.search(parsed.path + "/")
        return Resolved("link", m.group(1).upper() if m else None, _slug_title(parsed.path))
    compact = t.replace(" ", "")
    if _BARE_ASIN.match(compact.upper()) and re.search(r"\d", compact) and " " not in t \
            and (compact.upper().startswith("B0") or compact == compact.upper()):
        return Resolved("asin", compact.upper())
    return Resolved("name", None, t)
