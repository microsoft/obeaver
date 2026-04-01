from __future__ import annotations

from pathlib import Path


def test_version_uses_brand_markup(monkeypatch) -> None:
    import ofoundry.cli as cli
    from ofoundry._version import __version__
    from ofoundry.brand import COMMAND_STYLE, TEXT_STYLE

    printed: list[str] = []

    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: printed.append(str(args[0])))

    cli.version()

    assert printed == [f"[{COMMAND_STYLE}]ofoundry[/] [{TEXT_STYLE}]v{__version__}[/]"]


def test_banner_uses_brand_palette(monkeypatch, tmp_path: Path) -> None:
    import ofoundry.cli as cli
    from ofoundry.brand import RULE_STYLE, SUBTITLE_STYLE, TITLE_STYLE

    printed: list[tuple[object, object]] = []
    rules: list[str] = []

    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: printed.append((args, kwargs.get("style"))))
    monkeypatch.setattr(cli.console, "rule", lambda *args, **kwargs: rules.append(kwargs.get("style")))
    monkeypatch.setattr("ofoundry.config.get_models_dir", lambda: tmp_path)
    monkeypatch.setattr("ofoundry.config.get_ort_models_dir", lambda: tmp_path / "ort")
    monkeypatch.setattr("ofoundry.config.get_foundrylocal_models_dir", lambda: tmp_path / "foundrylocal")
    monkeypatch.setattr("ofoundry.config.get_ort_cache_dir", lambda: tmp_path / "cache_dir")

    cli._print_banner()

    assert rules == [RULE_STYLE, RULE_STYLE]
    assert any(args and args[0] == "Build local. Dam the cloud." and style == SUBTITLE_STYLE for args, style in printed)
    assert any(style == TITLE_STYLE for args, style in printed if args and isinstance(args[0], str) and args[0].strip())