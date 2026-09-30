"""Turning raw input (URLs, FQDNs, log fields) into the label the model scores."""

from __future__ import annotations

import tldextract

# Use the public suffix list snapshot bundled with tldextract; never fetch at runtime.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def normalize(raw: str) -> str:
    """Return the registrable label of a domain: ``"www.Example.co.uk."`` -> ``"example"``.

    DGAs randomise this part of the name, so it is what the model scores.
    Returns an empty string when no label can be extracted.
    """
    text = raw.strip().lower().rstrip(".")
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0].split(":", 1)[0]
    return _extract(text).domain
