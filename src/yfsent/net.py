from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request

USER_AGENT = "yfsent/0.1 (personal non-commercial research; RSS reader)"


class FetchError(RuntimeError):
    pass


def _verified_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    # Python 3.14/OpenSSL strict mode rejects some otherwise valid public-sector
    # certificates that omit legacy extension fields. CA and hostname checks stay on.
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def get_bytes(url: str, *, timeout: float = 15.0, retries: int = 2) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=_verified_ssl_context()
            ) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    raise FetchError(f"無法取得 {url}: {last_error}") from last_error
