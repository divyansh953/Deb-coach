"""Website delivery.

Posts the raw daily digest to the Next.js debate site so it can display the
latest brief. Mirrors the shape of delivery.telegram:

- send_to_website(final_doc) -> None

Design mirrors telegram.py on purpose:
- All env reads are lazy so DEV_MODE flips take effect mid-process.
- Failures are retried with backoff and then raised as
  WebsiteDeliveryError, but callers (main.run_daily) treat website
  delivery as best-effort so a site outage never blocks Telegram.

Overwrite model: the site stores exactly one "current" digest, so each
POST replaces the previous one. The 24h expiry lives on the site (a TTL
column / cleanup on read), not here.

Env vars:
- WEBSITE_DIGEST_URL     Full URL of the site's POST endpoint, e.g.
                         https://your-site.vercel.app/api/digest
- WEBSITE_DIGEST_SECRET  Shared secret sent as a Bearer token so only
                         this agent can publish.
"""
from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SEND_RETRIES = 2
SEND_BACKOFF_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 20


class WebsiteDeliveryError(RuntimeError):
    """Raised when the website POST fails after every retry."""


def _endpoint_url() -> str | None:
    url = os.getenv("WEBSITE_DIGEST_URL")
    return url.strip() if url else None


def _secret() -> str | None:
    secret = os.getenv("WEBSITE_DIGEST_SECRET")
    return secret.strip() if secret else None


def _dev_mode() -> bool:
    return os.getenv("DEV_MODE", "false").lower() == "true"


def send_to_website(final_doc: str) -> None:
    if not final_doc:
        return

    url = _endpoint_url()
    secret = _secret()

    if _dev_mode() or not url or not secret:
        preview = final_doc[:200]
        print(f"\n[Website DEV] Would POST digest ({len(final_doc)} chars) to {url or '<unset>'}\n{preview}\n{'=' * 40}")
        return

    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    payload = {"text": final_doc}

    last_error = ""
    for attempt in range(1, SEND_RETRIES + 2):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            print(f"[Website] Posted digest (attempt {attempt}, status {response.status_code})")
            return
        except Exception as exc:
            last_error = str(exc)
            print(f"[Website] Post failed (attempt {attempt}): {last_error[:300]}")
            if attempt <= SEND_RETRIES:
                time.sleep(SEND_BACKOFF_SECONDS * attempt)

    raise WebsiteDeliveryError(
        f"All {SEND_RETRIES + 1} website post attempts failed. Last error: {last_error}"
    )
