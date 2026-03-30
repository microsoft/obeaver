"""
Unit tests for ofoundry.cli (CLI command registration and help text).

Run with:
  pytest tests/test_cli.py -v
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ofoundry.cli import app

runner = CliRunner()


class TestCliHelp:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_mentions_ofoundry(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "ofoundry" in result.output.lower() or "llm" in result.output.lower()


class TestCliCommands:
    """Verify all expected subcommands are registered."""

    def test_run_command_registered(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower() or "run" in result.output.lower()

    def test_serve_command_registered(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "openai" in result.output.lower()

    def test_check_command_registered(self) -> None:
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0

    def test_version_command_registered(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_output(self) -> None:
        result = runner.invoke(app, ["version"])
        # Should contain a version-like string
        assert "." in result.output  # e.g. "0.1.0"

    def test_embed_command_registered(self) -> None:
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0

    def test_serve_embed_command_registered(self) -> None:
        result = runner.invoke(app, ["serve-embed", "--help"])
        assert result.exit_code == 0


class TestCliOptions:
    """Verify key options appear in help text."""

    def test_run_has_model_option(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert "--model" in result.output or "-m" in result.output

    def test_run_has_engine_option(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert "--engine" in result.output or "-E" in result.output

    def test_serve_has_host_option(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "--host" in result.output

    def test_serve_has_port_option(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "--port" in result.output


class TestIsVlModel:
    """Verify _is_vl_model detection logic."""

    def test_detects_vl_model_with_vision_onnx(self, tmp_path) -> None:
        from ofoundry.cli import _is_vl_model
        (tmp_path / "vision.onnx").touch()
        (tmp_path / "genai_config.json").touch()
        assert _is_vl_model(str(tmp_path)) is True

    def test_rejects_text_only_model(self, tmp_path) -> None:
        from ofoundry.cli import _is_vl_model
        (tmp_path / "model.onnx").touch()
        (tmp_path / "genai_config.json").write_text('{"model": {"type": "qwen2"}}')
        assert _is_vl_model(str(tmp_path)) is False

    def test_detects_vl_via_genai_config(self, tmp_path) -> None:
        from ofoundry.cli import _is_vl_model
        import json
        cfg = {"model": {"type": "qwen2_5_vl", "vision": {"filename": "vision.onnx"}}}
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg))
        assert _is_vl_model(str(tmp_path)) is True

    def test_rejects_nonexistent_path(self) -> None:
        from ofoundry.cli import _is_vl_model
        assert _is_vl_model("/nonexistent/path") is False

    def test_rejects_non_directory(self, tmp_path) -> None:
        from ofoundry.cli import _is_vl_model
        f = tmp_path / "somefile.txt"
        f.touch()
        assert _is_vl_model(str(f)) is False
