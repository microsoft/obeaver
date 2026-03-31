"""
Unit tests for obeaver.monitor (system resource monitoring).

Run with:
  pytest tests/test_monitor.py -v
"""

from __future__ import annotations

import pytest

from obeaver.monitor import MemoryInfo, get_all_memory, get_cpu_memory, get_process_memory


# ---------------------------------------------------------------------------
# MemoryInfo dataclass
# ---------------------------------------------------------------------------


class TestMemoryInfo:
    def test_all_fields(self) -> None:
        info = MemoryInfo(
            device="CPU", label="Test CPU", total_mb=16384.0,
            used_mb=8192.0, available_mb=8192.0, utilisation=0.5, status="active",
        )
        assert info.device == "CPU"
        assert info.label == "Test CPU"
        assert info.total_mb == 16384.0
        assert info.used_mb == 8192.0
        assert info.available_mb == 8192.0
        assert info.utilisation == 0.5
        assert info.status == "active"

    def test_as_dict(self) -> None:
        from dataclasses import asdict
        info = MemoryInfo(
            device="GPU",
            label="Test GPU",
            total_mb=8192.0,
            used_mb=4096.0,
            available_mb=4096.0,
            utilisation=0.5,
            status="active",
        )
        d = asdict(info)
        assert d["device"] == "GPU"
        assert d["label"] == "Test GPU"
        assert d["total_mb"] == 8192.0
        assert d["used_mb"] == 4096.0
        assert d["available_mb"] == 4096.0
        assert d["utilisation"] == 0.5
        assert d["status"] == "active"

    def test_asdict_returns_plain_dict(self) -> None:
        from dataclasses import asdict
        info = MemoryInfo(
            device="NPU", label="Test NPU", total_mb=0.0,
            used_mb=0.0, available_mb=0.0, utilisation=0.0, status="idle",
        )
        d = asdict(info)
        assert isinstance(d, dict)
        assert set(d.keys()) == {"device", "label", "total_mb", "used_mb", "available_mb", "utilisation", "status"}


# ---------------------------------------------------------------------------
# get_cpu_memory
# ---------------------------------------------------------------------------


class TestGetCpuMemory:
    def test_returns_memory_info(self) -> None:
        info = get_cpu_memory()
        assert isinstance(info, MemoryInfo)
        assert info.device == "CPU"

    def test_cpu_has_nonzero_total(self) -> None:
        info = get_cpu_memory()
        # Every machine has some RAM
        assert info.total_mb > 0

    def test_cpu_label_not_empty(self) -> None:
        info = get_cpu_memory()
        assert info.label != ""

    def test_cpu_status_is_active(self) -> None:
        info = get_cpu_memory()
        assert info.status == "active"

    def test_cpu_utilisation_in_range(self) -> None:
        info = get_cpu_memory()
        assert 0.0 <= info.utilisation <= 1.0


# ---------------------------------------------------------------------------
# get_process_memory
# ---------------------------------------------------------------------------


class TestGetProcessMemory:
    def test_returns_dict(self) -> None:
        info = get_process_memory()
        assert isinstance(info, dict)

    def test_has_required_keys(self) -> None:
        info = get_process_memory()
        assert "rss_mb" in info
        assert "vms_mb" in info
        assert "pid" in info

    def test_positive_values(self) -> None:
        info = get_process_memory()
        assert info["rss_mb"] > 0
        assert info["pid"] > 0


# ---------------------------------------------------------------------------
# get_all_memory
# ---------------------------------------------------------------------------


class TestGetAllMemory:
    def test_returns_dict(self) -> None:
        result = get_all_memory()
        assert isinstance(result, dict)

    def test_has_all_sections(self) -> None:
        result = get_all_memory()
        assert "cpu" in result
        assert "gpu" in result
        assert "npu" in result
        assert "process" in result
        assert "platform" in result

    def test_cpu_section_is_dict(self) -> None:
        result = get_all_memory()
        assert isinstance(result["cpu"], dict)
        assert result["cpu"]["device"] == "CPU"

    def test_platform_section(self) -> None:
        result = get_all_memory()
        p = result["platform"]
        assert "system" in p
        assert "machine" in p
        assert "python" in p
        assert isinstance(p["python"], str)

    def test_gpu_section_has_device(self) -> None:
        result = get_all_memory()
        assert result["gpu"]["device"] == "GPU"

    def test_npu_section_has_device(self) -> None:
        result = get_all_memory()
        assert result["npu"]["device"] == "NPU"

    def test_json_serialisable(self) -> None:
        import json
        result = get_all_memory()
        # Should not raise
        serialised = json.dumps(result)
        assert isinstance(serialised, str)
