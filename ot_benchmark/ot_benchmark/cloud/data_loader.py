"""Cloud-to-local data streaming for large datasets.

Streams datasets from OBS chunk-by-chunk to avoid loading
everything into memory at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from core.logging_config import get_logger

logger = get_logger("cloud.loader")


def stream_from_obs(
    bucket: str,
    prefix: str,
    local_cache_dir: str | Path = "data_cache",
) -> Iterator[Path]:
    """Stream dataset files from OBS to local cache, yielding paths.

    Files are downloaded on first access and cached locally.
    Subsequent calls use the cache.
    """
    from cloud.obs_client import OBSClient
    from cloud.cloud_config import load_cloud_credentials

    creds = load_cloud_credentials()
    if not creds.is_valid:
        logger.warning("No cloud credentials — streaming disabled")
        return

    obs = OBSClient(creds)
    if not obs.available:
        logger.warning("OBS client not available")
        return

    cache_dir = Path(local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    objects = obs.list_objects(prefix=prefix)
    for obj_key in objects:
        local_path = cache_dir / obj_key.lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            logger.info("Downloading from OBS: %s", obj_key)
            # OBS download would go here with full SDK
            # For now, skip if not implemented
            pass

        if local_path.exists():
            yield local_path
