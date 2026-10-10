"""Huawei Cloud OBS (Object Storage Service) client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cloud.cloud_config import CloudCredentials
from core.logging_config import get_logger

logger = get_logger("cloud.obs")


class OBSClient:
    """Minimal OBS client for uploading benchmark results and downloading datasets.

    Requires esdk-obs-python: pip install esdk-obs-python
    """

    def __init__(self, credentials: CloudCredentials):
        self._creds = credentials
        self._client = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """Check whether OBS is configured and importable."""
        if not self._creds.is_valid:
            return False
        try:
            from obs import ObsClient
            return True
        except ImportError:
            return False

    def _ensure_client(self):
        """Lazily initialize the OBS client."""
        if self._initialized:
            return
        if not self.available:
            raise RuntimeError("OBS not available: missing credentials or esdk-obs-python")
        try:
            from obs import ObsClient
            self._client = ObsClient(
                access_key_id=self._creds.access_key,
                secret_access_key=self._creds.secret_key,
                server=self._creds.endpoint,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "esdk-obs-python is required for OBS. Install: pip install esdk-obs-python"
            )

    def upload_file(
        self,
        local_path: str | Path,
        object_key: str | None = None,
    ) -> bool:
        """Upload a single file to OBS.

        Args:
            local_path: Path to the local file.
            object_key: OBS object key. Defaults to local filename.

        Returns:
            True if upload succeeded.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("File not found: %s", local_path)
            return False

        if object_key is None:
            object_key = local_path.name

        self._ensure_client()

        try:
            resp = self._client.putFile(
                bucketName=self._creds.bucket,
                objectKey=object_key,
                file_path=str(local_path),
            )
            if resp.status < 300:
                logger.info("Uploaded: %s -> obs://%s/%s",
                            local_path.name, self._creds.bucket, object_key)
                return True
            else:
                logger.error("Upload failed: %s (status %d)", object_key, resp.status)
                return False
        except Exception as e:
            logger.error("Upload error for %s: %s", object_key, e)
            return False

    def upload_directory(
        self,
        local_dir: str | Path,
        prefix: str = "",
    ) -> int:
        """Upload all files in a directory to OBS.

        Args:
            local_dir: Local directory path.
            prefix: OBS object key prefix (folder path).

        Returns:
            Number of files successfully uploaded.
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            logger.warning("Directory not found: %s", local_dir)
            return 0

        count = 0
        for root, _, files in os.walk(local_dir):
            for fname in files:
                fpath = Path(root) / fname
                rel = fpath.relative_to(local_dir)
                object_key = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
                if self.upload_file(fpath, object_key):
                    count += 1
        return count

    def list_objects(self, prefix: str = "") -> list[str]:
        """List object keys in the configured bucket.

        Args:
            prefix: Optional key prefix filter.

        Returns:
            List of object keys.
        """
        self._ensure_client()
        try:
            resp = self._client.listObjects(
                bucketName=self._creds.bucket,
                prefix=prefix,
            )
            if resp.status < 300 and resp.body.contents:
                return [obj.key for obj in resp.body.contents]
            return []
        except Exception as e:
            logger.error("List objects error: %s", e)
            return []
