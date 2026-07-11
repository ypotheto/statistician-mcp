from __future__ import annotations

import io
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import pandas as pd

from statistician_mcp.errors import ValidationError

MAX_URL_BYTES = 50 * 1024 * 1024


async def fetch_tabular_from_url(url: str) -> pd.DataFrame:
    """Fetch a CSV/Excel/Parquet file from an https URL and parse it into a DataFrame.

    Guards against SSRF by requiring https and rejecting hostnames that resolve to
    private/loopback/link-local addresses, and disables redirect-following so a public
    URL can't bounce the request to an internal one. This is a baseline, not a full
    anti-DNS-rebinding defense (no connect-time IP pinning) — see Phase 7 hardening.
    """
    _assert_public_https_url(url)

    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        try:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise ValidationError(f"fetching URL failed with status {response.status_code}")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_URL_BYTES:
                        raise ValidationError(
                            f"remote file exceeds the {MAX_URL_BYTES // (1024 * 1024)} MB limit"
                        )
                content_type = response.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            raise ValidationError(f"could not fetch URL: {exc}") from exc

    return _parse_tabular(url, content_type, bytes(content))


def _assert_public_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValidationError("only https URLs are allowed")
    host = parsed.hostname
    if not host:
        raise ValidationError("URL is missing a host")
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise ValidationError(f"could not resolve host '{host}'") from exc
    for addr_info in addr_infos:
        ip = ipaddress.ip_address(addr_info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValidationError(
                f"host '{host}' resolves to a non-public address and is not allowed"
            )


def _parse_tabular(url: str, content_type: str, data: bytes) -> pd.DataFrame:
    lowered_url = url.lower()
    try:
        if "parquet" in content_type or lowered_url.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        if "spreadsheet" in content_type or lowered_url.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(data))
        return pd.read_csv(io.StringIO(data.decode("utf-8")))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"could not parse fetched file as tabular data: {exc}") from exc
