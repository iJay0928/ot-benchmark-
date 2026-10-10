"""Huawei Cloud integration for OT Benchmark."""

from cloud.cloud_config import CloudCredentials, load_cloud_credentials
from cloud.obs_client import OBSClient

__all__ = ["CloudCredentials", "load_cloud_credentials", "OBSClient"]
