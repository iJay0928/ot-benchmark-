"""YAML/JSON configuration system for the OT Benchmark system."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from .exceptions import ConfigValidationError
from .logging_config import get_logger

logger = get_logger("config")


@dataclass
class WeightConfigData:
    """Weight configuration data (decoupled from benchmark_engine)."""
    w_accuracy: float = 0.35
    w_speed: float = 0.30
    w_memory: float = 0.25
    w_success: float = 0.10
    max_runtime: float = 120.0
    max_memory: float = 8192.0

    def __post_init__(self):
        total = self.w_accuracy + self.w_speed + self.w_memory + self.w_success
        if total <= 0:
            raise ConfigValidationError("权重总和必须大于 0")
        self.w_accuracy /= total
        self.w_speed /= total
        self.w_memory /= total
        self.w_success /= total


@dataclass
class ExecutionConfig:
    """Execution parameters."""
    parallel_workers: int = 0  # 0 = auto
    timeout_per_algorithm: float = 300.0
    memory_limit_mb: float = 8192.0


@dataclass
class BackendConfig:
    """Compute backend configuration."""
    type: str = "auto"  # auto | numpy | torch | cupy
    gpu_id: int = 0
    fallback: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    charts: bool = True
    html_report: bool = True
    format: str = "png"
    dpi: int = 150
    output_dir: str = "benchmark_results"


@dataclass
class CloudConfig:
    """Cloud service configuration."""
    obs_enabled: bool = False
    obs_endpoint: str = "https://obs.cn-south-1.myhuaweicloud.com"
    obs_bucket: str = "ot-benchmark-results"
    modelarts_enabled: bool = False


@dataclass
class LoggingConfigData:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Master configuration for the OT Benchmark system."""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    weight_config: WeightConfigData = field(default_factory=WeightConfigData)
    output: OutputConfig = field(default_factory=OutputConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    logging: LoggingConfigData = field(default_factory=LoggingConfigData)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load configuration from a YAML file.

        Falls back to JSON if yaml is not importable.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigValidationError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                import yaml
                data = yaml.safe_load(f) or {}
            except ImportError:
                logger.debug("yaml not available, trying JSON")
                f.seek(0)
                data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Build config from a nested dictionary."""
        return cls(
            execution=ExecutionConfig(**data.get("execution", {})),
            backend=BackendConfig(**data.get("backend", {})),
            weight_config=WeightConfigData(**data.get("weight_config", {})),
            output=OutputConfig(**data.get("output", {})),
            cloud=CloudConfig(**data.get("cloud", {})),
            logging=LoggingConfigData(**data.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "execution": self.execution.__dict__,
            "backend": self.backend.__dict__,
            "weight_config": self.weight_config.__dict__,
            "output": self.output.__dict__,
            "cloud": self.cloud.__dict__,
            "logging": self.logging.__dict__,
        }

    def merge_cli_overrides(self, **kwargs: Any) -> BenchmarkConfig:
        """Override config fields from CLI arguments (non-None values only)."""
        for key, value in kwargs.items():
            if value is None:
                continue
            parts = key.split(".")
            if len(parts) == 1:
                if hasattr(self, parts[0]):
                    setattr(self, parts[0], value)
            elif len(parts) == 2:
                section, attr = parts
                if hasattr(self, section):
                    obj = getattr(self, section)
                    if hasattr(obj, attr):
                        setattr(obj, attr, value)
        return self


def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    return Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(config_path: Optional[str | Path] = None) -> BenchmarkConfig:
    """Load benchmark configuration.

    Priority: specified path > default.yaml > hardcoded defaults.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise ConfigValidationError(f"Config file not found: {path}")
        logger.info("Using config file: %s", path)
        return BenchmarkConfig.from_yaml(path)

    default_path = get_default_config_path()
    if default_path.exists():
        logger.info("Using default config: %s", default_path)
        return BenchmarkConfig.from_yaml(default_path)

    logger.info("No config file found, using hardcoded defaults")
    return BenchmarkConfig()
