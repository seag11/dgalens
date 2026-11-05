"""Turning raw input (URLs, FQDNs, log fields) into the label the model scores."""

from __future__ import annotations

import ipaddress

import tldextract

# Use the public suffix list snapshot bundled with tldextract; never fetch at runtime.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def normalize(raw: str) -> str:
    """Return the registrable label of a domain: ``"www.Example.co.uk."`` -> ``"example"``.

    DGAs randomise this part of the name, so it is what the model scores.
    Returns an empty string when no label can be extracted, including for IP
    addresses, which have no label and would otherwise score as gibberish.
    """
    text = raw.strip().lower().rstrip(".")
    if "://" in text:
        text = text.split("://", 1)[1]
    host = text.split("/", 1)[0]
    if host.startswith("[") and "]" in host:  # [IPv6] or [IPv6]:port
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:  # host:port; more colons means a bare IPv6 address
        host = host.split(":", 1)[0]
    if _is_ip_address(host):
        return ""
    return _extract(host).domain


def _is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True
