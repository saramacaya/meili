"""Cloudflare R2 storage for Meili's map tiles (S3-compatible).

Configured with four environment variables (Render and GitHub secrets):

  R2_ACCOUNT_ID         Cloudflare account ID
  R2_ACCESS_KEY_ID      R2 API token access key
  R2_SECRET_ACCESS_KEY  R2 API token secret
  R2_BUCKET             bucket name (default: meili-tiles)

If any of the first three is missing, client() returns None and callers fall
back to Supabase Storage, so the backend keeps working before R2 is set up.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_lock = threading.Lock()
_client = None
_checked = False


def bucket() -> str:
    return os.getenv("R2_BUCKET", "meili-tiles")


def client():
    """A cached boto3 S3 client for R2, or None when R2 isn't configured."""
    global _client, _checked
    if _checked:
        return _client
    with _lock:
        if _checked:
            return _client
        account = os.getenv("R2_ACCOUNT_ID")
        key_id = os.getenv("R2_ACCESS_KEY_ID")
        secret = os.getenv("R2_SECRET_ACCESS_KEY")
        if account and key_id and secret:
            try:
                import boto3
                from botocore.config import Config

                _client = boto3.client(
                    "s3",
                    endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
                    aws_access_key_id=key_id,
                    aws_secret_access_key=secret,
                    region_name="auto",
                    config=Config(
                        retries={"max_attempts": 5, "mode": "adaptive"},
                        max_pool_connections=32,
                        connect_timeout=10,
                        read_timeout=60,
                    ),
                )
            except Exception as error:  # noqa: BLE001
                print(f"R2 is configured but could not start: {error}")
                _client = None
        _checked = True
        return _client


def get_bytes(path: str) -> Optional[bytes]:
    """The object's bytes, or None if it doesn't exist or the read failed."""
    c = client()
    if c is None:
        return None
    try:
        return c.get_object(Bucket=bucket(), Key=path)["Body"].read()
    except Exception:  # noqa: BLE001  (NoSuchKey, network errors)
        return None


def put_bytes(path: str, body: bytes, content_type: str = "application/gzip") -> bool:
    c = client()
    if c is None:
        return False
    try:
        c.put_object(Bucket=bucket(), Key=path, Body=body, ContentType=content_type)
        return True
    except Exception as error:  # noqa: BLE001
        print(f"R2 upload failed for {path}: {error}")
        return False


def list_keys(prefix: str) -> list[str]:
    c = client()
    if c is None:
        return []
    keys: list[str] = []
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys
