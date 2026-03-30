"""
Unit tests for ofoundry.config (ServerConfig dataclass).

Run with:
  pytest tests/test_config.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ofoundry.config import ServerConfig


class TestServerConfigDefaults:
    def test_default_host(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"

    def test_default_port(self) -> None:
        cfg = ServerConfig()
        assert cfg.port == 1573

    def test_default_execution_provider(self) -> None:
        cfg = ServerConfig()
        assert cfg.execution_provider == "cpu"

    def test_default_generation_params(self) -> None:
        cfg = ServerConfig()
        assert cfg.default_max_new_tokens == 1024
        assert cfg.default_temperature == 1.0
        assert cfg.default_top_p == 0.95
        assert cfg.default_top_k == 50

    def test_default_system_prompt(self) -> None:
        cfg = ServerConfig()
        assert "helpful" in cfg.default_system_prompt.lower()


class TestServerConfigValidation:
    def test_validate_with_existing_path(self, tmp_path: Path) -> None:
        cfg = ServerConfig(model_path=tmp_path)
        # Should not raise
        cfg.validate()

    def test_validate_with_nonexistent_path(self) -> None:
        cfg = ServerConfig(model_path=Path("/nonexistent/path/model"))
        with pytest.raises(ValueError, match="model_path does not exist"):
            cfg.validate()

    def test_validate_with_invalid_ep(self, tmp_path: Path) -> None:
        cfg = ServerConfig(model_path=tmp_path, execution_provider="tpu")
        with pytest.raises(ValueError, match="Unsupported execution_provider"):
            cfg.validate()

    def test_validate_accepts_cuda(self, tmp_path: Path) -> None:
        cfg = ServerConfig(model_path=tmp_path, execution_provider="cuda")
        cfg.validate()

    def test_validate_accepts_dml(self, tmp_path: Path) -> None:
        cfg = ServerConfig(model_path=tmp_path, execution_provider="dml")
        cfg.validate()


class TestServerConfigCustom:
    def test_custom_values(self, tmp_path: Path) -> None:
        cfg = ServerConfig(
            model_path=tmp_path,
            host="0.0.0.0",
            port=9000,
            execution_provider="cuda",
            default_max_new_tokens=2048,
        )
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.execution_provider == "cuda"
        assert cfg.default_max_new_tokens == 2048
