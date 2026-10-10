"""Cloud credential management — never hardcode credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger

logger = get_logger("cloud")


@dataclass
class CloudCredentials:
    """Huawei Cloud credentials loaded from environment or config files."""

    access_key: str = ""
    secret_key: str = ""
    endpoint: str = "https://obs.cn-south-1.myhuaweicloud.com"
    bucket: str = "ot-benchmark-results"
    project_id: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.access_key and self.secret_key)


def _load_from_env() -> Optional[CloudCredentials]:
    """Try loading credentials from environment variables."""
    ak = os.environ.get("HW_ACCESS_KEY") or os.environ.get("HUAWEI_ACCESS_KEY")
    sk = os.environ.get("HW_SECRET_KEY") or os.environ.get("HUAWEI_SECRET_KEY")
    if ak and sk:
        return CloudCredentials(
            access_key=ak,
            secret_key=sk,
            endpoint=os.environ.get("HW_OBS_ENDPOINT", "https://obs.cn-south-1.myhuaweicloud.com"),
            bucket=os.environ.get("HW_OBS_BUCKET", "ot-benchmark-results"),
            project_id=os.environ.get("HW_PROJECT_ID", ""),
        )
    return None


def _load_from_file() -> Optional[CloudCredentials]:
    """Try loading credentials from ~/.huawei/credentials."""
    cred_path = Path.home() / ".huawei" / "credentials"
    if not cred_path.exists():
        return None

    data: dict[str, str] = {}
    with open(cred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip().strip('"').strip("'")

    if "access_key" in data and "secret_key" in data:
        return CloudCredentials(
            access_key=data["access_key"],
            secret_key=data["secret_key"],
            endpoint=data.get("endpoint", "https://obs.cn-south-1.myhuaweicloud.com"),
            bucket=data.get("bucket", "ot-benchmark-results"),
            project_id=data.get("project_id", ""),
        )
    return None


def load_cloud_credentials() -> CloudCredentials:
    """Load cloud credentials in priority order.

    Priority: env vars > ~/.huawei/credentials > empty (cloud disabled).
    """
    creds = _load_from_env()
    if creds and creds.is_valid:
        logger.info("Cloud credentials loaded from environment variables")
        return creds

    creds = _load_from_file()
    if creds and creds.is_valid:
        logger.info("Cloud credentials loaded from ~/.huawei/credentials")
        return creds

    logger.info("No cloud credentials found — cloud features disabled")
    return CloudCredentials()
