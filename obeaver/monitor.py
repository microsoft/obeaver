"""
System resource monitoring for obeaver.

Provides CPU, GPU, and NPU memory usage information for the dashboard UI.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryInfo:
    """Memory usage for a single compute device."""

    device: str          # "CPU", "GPU", "NPU"
    label: str           # human-readable device name
    total_mb: float      # total memory in MiB
    used_mb: float       # used memory in MiB
    available_mb: float  # available memory in MiB
    utilisation: float   # 0.0 to 1.0
    status: str          # "active", "idle", "unavailable"


def get_cpu_memory() -> MemoryInfo:
    """Return current system RAM usage."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return MemoryInfo(
            device="CPU",
            label=_cpu_label(),
            total_mb=vm.total / (1024 * 1024),
            used_mb=vm.used / (1024 * 1024),
            available_mb=vm.available / (1024 * 1024),
            utilisation=vm.percent / 100.0,
            status="active",
        )
    except ImportError:
        return _cpu_memory_fallback()


def _cpu_label() -> str:
    """Return a short label for the CPU."""
    return platform.processor() or platform.machine() or "System CPU"


def _cpu_memory_fallback() -> MemoryInfo:
    """Fallback CPU memory reading without psutil."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info: dict[str, int] = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # in kB
            total = info.get("MemTotal", 0) / 1024
            available = info.get("MemAvailable", info.get("MemFree", 0)) / 1024
            used = total - available
            return MemoryInfo(
                device="CPU",
                label=_cpu_label(),
                total_mb=total,
                used_mb=used,
                available_mb=available,
                utilisation=used / total if total > 0 else 0,
                status="active",
            )
        except OSError:
            pass
    return MemoryInfo(
        device="CPU",
        label=_cpu_label(),
        total_mb=0, used_mb=0, available_mb=0,
        utilisation=0, status="unavailable",
    )


def get_gpu_memory() -> MemoryInfo:
    """Return GPU memory usage (NVIDIA via nvidia-smi, or DirectML on Windows)."""
    # Try NVIDIA first
    info = _nvidia_gpu_memory()
    if info:
        return info

    # Try AMD ROCm
    info = _rocm_gpu_memory()
    if info:
        return info

    # Try DirectML / Windows GPU
    if sys.platform == "win32":
        info = _windows_gpu_memory()
        if info:
            return info

    return MemoryInfo(
        device="GPU",
        label="No GPU detected",
        total_mb=0, used_mb=0, available_mb=0,
        utilisation=0, status="unavailable",
    )


def _nvidia_gpu_memory() -> Optional[MemoryInfo]:
    """Query NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name = parts[0]
                total = float(parts[1])
                used = float(parts[2])
                free = float(parts[3])
                util = float(parts[4]) / 100.0
                return MemoryInfo(
                    device="GPU",
                    label=name,
                    total_mb=total,
                    used_mb=used,
                    available_mb=free,
                    utilisation=util,
                    status="active" if used > 0 else "idle",
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _rocm_gpu_memory() -> Optional[MemoryInfo]:
    """Query AMD GPU via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return MemoryInfo(
                device="GPU",
                label="AMD GPU (ROCm)",
                total_mb=0, used_mb=0, available_mb=0,
                utilisation=0, status="idle",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _windows_gpu_memory() -> Optional[MemoryInfo]:
    """Query GPU on Windows using PowerShell / WMI."""
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -First 1 Name, AdapterRAM | "
                "ForEach-Object { $_.Name + ',' + $_.AdapterRAM }",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip() if parts else "Windows GPU"
            total_bytes = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
            total_mb = total_bytes / (1024 * 1024)
            return MemoryInfo(
                device="GPU",
                label=name,
                total_mb=total_mb,
                used_mb=0,
                available_mb=total_mb,
                utilisation=0,
                status="idle",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def get_npu_memory() -> MemoryInfo:
    """Return NPU memory/status information."""
    # Check Intel NPU
    info = _intel_npu_info()
    if info:
        return info

    # Check Qualcomm NPU (Windows on ARM)
    info = _qualcomm_npu_info()
    if info:
        return info

    return MemoryInfo(
        device="NPU",
        label="No NPU detected",
        total_mb=0, used_mb=0, available_mb=0,
        utilisation=0, status="unavailable",
    )


def _intel_npu_info() -> Optional[MemoryInfo]:
    """Check for Intel NPU availability."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-PnpDevice -FriendlyName '*NPU*','*Neural*','*AI Boost*' "
                    "-ErrorAction SilentlyContinue | "
                    "Where-Object { $_.Status -eq 'OK' } | "
                    "Select-Object -First 1 FriendlyName | "
                    "ForEach-Object { $_.FriendlyName }",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                name = result.stdout.strip()
                return MemoryInfo(
                    device="NPU",
                    label=name,
                    total_mb=0,
                    used_mb=0,
                    available_mb=0,
                    utilisation=0,
                    status="idle",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    # Check for Intel NPU via OpenVINO
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "NPU" in devices:
            return MemoryInfo(
                device="NPU",
                label="Intel NPU (OpenVINO)",
                total_mb=0, used_mb=0, available_mb=0,
                utilisation=0, status="idle",
            )
    except (ImportError, Exception):
        pass

    return None


def _qualcomm_npu_info() -> Optional[MemoryInfo]:
    """Check for Qualcomm NPU (Snapdragon X Elite, etc.)."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-PnpDevice -FriendlyName '*Qualcomm*NPU*','*Hexagon*' "
                    "-ErrorAction SilentlyContinue | "
                    "Where-Object { $_.Status -eq 'OK' } | "
                    "Select-Object -First 1 FriendlyName | "
                    "ForEach-Object { $_.FriendlyName }",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                name = result.stdout.strip()
                return MemoryInfo(
                    device="NPU",
                    label=name,
                    total_mb=0, used_mb=0, available_mb=0,
                    utilisation=0, status="idle",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    return None


def get_process_memory() -> dict:
    """Return memory used by the current obeaver process."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
            "pid": os.getpid(),
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0, "pid": os.getpid()}


import threading

# Cache GPU and NPU info — these don't change at runtime and the
# PowerShell/WMI queries are very slow (up to 10s each).
_gpu_cache: Optional[MemoryInfo] = None
_npu_cache: Optional[MemoryInfo] = None
_cache_lock = threading.Lock()


def get_all_memory() -> dict:
    """Return a summary of all device memory as a JSON-serialisable dict."""
    global _gpu_cache, _npu_cache

    cpu = get_cpu_memory()
    proc = get_process_memory()

    # GPU and NPU info is cached — the slow PowerShell queries run only once.
    if _gpu_cache is None:
        with _cache_lock:
            if _gpu_cache is None:
                _gpu_cache = get_gpu_memory()
    if _npu_cache is None:
        with _cache_lock:
            if _npu_cache is None:
                _npu_cache = get_npu_memory()

    gpu = _gpu_cache
    npu = _npu_cache

    def _to_dict(m: MemoryInfo) -> dict:
        return {
            "device": m.device,
            "label": m.label,
            "total_mb": round(m.total_mb, 1),
            "used_mb": round(m.used_mb, 1),
            "available_mb": round(m.available_mb, 1),
            "utilisation": round(m.utilisation, 4),
            "status": m.status,
        }

    return {
        "cpu": _to_dict(cpu),
        "gpu": _to_dict(gpu),
        "npu": _to_dict(npu),
        "process": proc,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
