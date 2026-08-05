"""Optional S3/MinIO publishing for dataset snapshots (M2).

**Role.** The local content-addressed snapshot directory is the source of truth.
This module copies one into the staging bucket the pipeline team pulls from
(Q-C: this module stages, they import into their own MLflow - we never write it).

**Deliberately opt-in and deliberately unable to fail an export.** By the time
publishing runs, the snapshot is already written, hashed and recorded. A
misconfigured bucket or an unreachable MinIO must not turn that into a failed
export - it reports `published: false` with the reason instead. Retrying is a
separate call against an already-immutable snapshot, which is safe precisely
because the snapshot cannot have changed in the meantime.

boto3 rather than the `minio` SDK: MinIO speaks S3, and boto3 means pointing at
real S3 later is an endpoint change rather than a rewrite. It is imported lazily
so a deployment that never publishes does not need it installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PublishResult:
    published: bool
    uploaded_files: int = 0
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "published": self.published,
            "uploaded_files": self.uploaded_files,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "error": self.error,
        }


def is_configured(settings: Settings) -> bool:
    return bool(settings.s3_endpoint_url and settings.s3_access_key and settings.s3_secret_key)


def _client(settings: Settings):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        # Path-style addressing: MinIO does not do virtual-host-style buckets
        # unless specifically set up for it, and a bucket name with dots breaks
        # virtual-host TLS anyway.
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
    )


def publish_snapshot(settings: Settings, snapshot_dir: Path, snapshot_id: str) -> PublishResult:
    """Upload a finalized snapshot directory under `<bucket>/<prefix><snapshot_id>/`.

    Never raises. Every failure path returns a PublishResult carrying the reason,
    because the caller has already committed a valid snapshot to disk and the
    only useful thing to do with an upload failure is report it.

    Existing objects are skipped rather than re-uploaded: a snapshot is immutable,
    so an object already present at the same key has identical content, and
    re-uploading it would only cost bandwidth. That also makes a retry after a
    partial upload resume rather than start over.
    """
    if not is_configured(settings):
        return PublishResult(
            published=False,
            error="Object store not configured (set S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY)",
        )

    bucket = settings.s3_bucket
    prefix = settings.s3_prefix.strip("/")
    key_root = f"{prefix}/{snapshot_id.replace('sha256:', 'sha256-')}" if prefix else snapshot_id.replace(
        "sha256:", "sha256-"
    )

    try:
        client = _client(settings)
    except ImportError as exc:
        return PublishResult(
            published=False,
            bucket=bucket,
            error=f"boto3 not installed, cannot publish: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - reporting beats propagating here
        logger.exception("Could not build an S3 client")
        return PublishResult(published=False, bucket=bucket, error=f"{type(exc).__name__}: {exc}")

    try:
        _ensure_bucket(client, bucket)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Bucket %s unusable", bucket)
        return PublishResult(published=False, bucket=bucket, error=f"{type(exc).__name__}: {exc}")

    existing = _existing_keys(client, bucket, key_root)

    uploaded = 0
    try:
        for path in sorted(p for p in snapshot_dir.rglob("*") if p.is_file()):
            key = f"{key_root}/{path.relative_to(snapshot_dir).as_posix()}"
            if key in existing:
                continue
            client.upload_file(str(path), bucket, key)
            uploaded += 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload of snapshot %s failed partway", snapshot_id)
        return PublishResult(
            published=False,
            uploaded_files=uploaded,
            bucket=bucket,
            prefix=key_root,
            error=f"{type(exc).__name__}: {exc} (partial upload; retry is safe and will resume)",
        )

    logger.info("Published snapshot %s to %s/%s (%d new object(s))", snapshot_id, bucket, key_root, uploaded)
    return PublishResult(published=True, uploaded_files=uploaded, bucket=bucket, prefix=key_root)


def freeze_golden_item(
    settings: Settings,
    *,
    dataset_view: str,
    version: int,
    image_id: str,
    image_path: Path,
    label_text: str,
) -> PublishResult:
    """Upload one golden-set item's image + label bytes to the **separate**
    golden bucket (M4), under `<golden_prefix>/<dataset_view>/v<version>/`.

    Deliberately its own bucket (settings.golden_s3_bucket), not a prefix
    under settings.s3_bucket - D-Q4 / build plan Q-C require structurally
    separate storage, because the only contamination mitigation that counts
    is that no export/propagation/triage path can write here at all. A
    shared bucket with a "golden/" prefix would still be one IAM boundary an
    export path could accidentally reach into; a separate bucket cannot be,
    short of explicitly being handed its credentials.

    Same never-raises/best-effort contract as publish_snapshot: freezing to
    object storage is a nice-to-have copy for the pipeline team to pull from.
    The load-bearing exclusion (an image_id can never end up in a train/val
    export once it's golden) lives in the golden_set_items DB row, written
    by the caller regardless of whether this upload succeeds - see
    golden_service.add_items.
    """
    if not is_configured(settings):
        return PublishResult(
            published=False,
            error="Object store not configured (set S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY)",
        )

    bucket = settings.golden_s3_bucket
    prefix = settings.golden_s3_prefix.strip("/")
    key_root = f"{prefix}/{dataset_view}/v{version}" if prefix else f"{dataset_view}/v{version}"

    try:
        client = _client(settings)
    except ImportError as exc:
        return PublishResult(published=False, bucket=bucket, error=f"boto3 not installed, cannot publish: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not build an S3 client")
        return PublishResult(published=False, bucket=bucket, error=f"{type(exc).__name__}: {exc}")

    try:
        _ensure_bucket(client, bucket)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Golden bucket %s unusable", bucket)
        return PublishResult(published=False, bucket=bucket, error=f"{type(exc).__name__}: {exc}")

    image_key = f"{key_root}/images/{image_path.name}"
    label_key = f"{key_root}/labels/{image_id}.txt"
    try:
        client.upload_file(str(image_path), bucket, image_key)
        client.put_object(Bucket=bucket, Key=label_key, Body=label_text.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Freezing golden item %s (%s v%d) failed", image_id, dataset_view, version)
        return PublishResult(published=False, bucket=bucket, prefix=key_root, error=f"{type(exc).__name__}: {exc}")

    logger.info("Froze golden item %s to %s/%s", image_id, bucket, image_key)
    return PublishResult(published=True, uploaded_files=2, bucket=bucket, prefix=image_key)


def _ensure_bucket(client, bucket: str) -> None:
    """Create the bucket if it isn't there. MinIO starts with none, and failing
    a publish because nobody clicked "create bucket" in a console is not a
    useful failure."""
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in (403, 404):
            raise
        if status == 403:
            # Exists but this key can't inspect it; a write may still succeed, so
            # don't pre-emptively fail here.
            logger.warning("No permission to head bucket %s; attempting upload anyway", bucket)
            return
    client.create_bucket(Bucket=bucket)
    logger.info("Created bucket %s", bucket)


def _existing_keys(client, bucket: str, key_root: str) -> set[str]:
    """Keys already under this snapshot's prefix, so a retry resumes. An error
    here degrades to "assume nothing is uploaded" rather than failing - the worst
    case is re-uploading bytes, which is safe because the content is identical."""
    keys: set[str] = set()
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{key_root}/"):
            for obj in page.get("Contents", []) or []:
                keys.add(obj["Key"])
    except Exception:  # noqa: BLE001
        logger.warning("Could not list existing objects under %s/%s; will upload all", bucket, key_root)
    return keys
