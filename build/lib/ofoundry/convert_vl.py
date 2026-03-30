"""VL (Vision-Language) model conversion using Olive optimization pipeline.

Follows the patterns from microsoft/olive-recipes to export and optimize
vision-language models into ONNX format with INT4 quantization.

The pipeline exports three sub-models:
  - Text decoder (via onnxruntime-genai ModelBuilder, called directly)
  - Vision encoder (via Olive: Dynamo export + graph optimizations + INT4)
  - Text embedding (via Olive: ONNX conversion + graph optimizations + INT4)

The text sub-model is built directly using the onnxruntime-genai builder API
because onnxruntime-genai <= 0.12.x does not yet support Qwen3VL in its
create_model() dispatcher.  We reuse the Qwen2.5-VL text model builder and
add QK-normalization for Qwen3.

Reference:
  https://github.com/microsoft/olive-recipes
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OLIVE_RECIPES_REPO = "https://github.com/microsoft/olive-recipes.git"
ORT_GENAI_REPO = "https://github.com/microsoft/onnxruntime-genai.git"

# Known VL model recipes: HF model name → olive-recipes subdirectory
VL_RECIPES: dict[str, str] = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen-Qwen3-VL-2B-Instruct/builtin",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen-Qwen2.5-VL-3B-Instruct/builtin",
}


def find_recipe(model_name: str) -> str | None:
    """Find the olive-recipes subdirectory for a VL model."""
    if model_name in VL_RECIPES:
        return VL_RECIPES[model_name]
    short = model_name.split("/")[-1]
    for key, val in VL_RECIPES.items():
        if short in key:
            return val
    return None


# ---------------------------------------------------------------------------
# Build onnxruntime-genai from source
# ---------------------------------------------------------------------------

def _detect_build_config(device: str) -> dict:
    """Detect the OS and return the build command args for onnxruntime-genai.

    Returns a dict with keys:
      - os_name: "Windows" | "Linux" | "Darwin"
      - build_args: list of CLI args for build.py
      - wheel_glob: glob pattern to find the built wheel
    """
    os_name = platform.system()  # "Windows", "Linux", "Darwin"

    # We only need the Python wheel — skip examples and tests to avoid
    # missing-header errors (examples require a separate ORT include dir).
    build_args = ["--config", "Release", "--skip_examples", "--skip_tests"]

    if os_name == "Windows":
        if device in ("cuda", "gpu"):
            # Try to find CUDA_HOME from environment
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                build_args += ["--use_trt_rtx", "--cuda_home", cuda_home]
            else:
                build_args += ["--use_cuda"]
        elif device == "dml":
            build_args += ["--use_dml"]
        # else: CPU build (default args)
        wheel_glob = "build/Windows/Release/wheel/*.whl"

    elif os_name == "Linux":
        if device in ("cuda", "gpu"):
            build_args += ["--use_cuda"]
        # else: CPU build (default args)
        wheel_glob = "build/Linux/Release/wheel/*.whl"

    elif os_name == "Darwin":
        # macOS: only CPU build supported
        wheel_glob = "build/macOS/Release/wheel/*.whl"

    else:
        raise RuntimeError(f"Unsupported OS: {os_name}")

    return {
        "os_name": os_name,
        "build_args": build_args,
        "wheel_glob": wheel_glob,
    }


def _install_cmake() -> None:
    """Attempt to install cmake automatically based on the current OS."""
    os_name = platform.system()
    print("[build-from-source] cmake not found – attempting automatic install...")

    if os_name == "Darwin":
        if shutil.which("brew") is None:
            raise RuntimeError(
                "cmake is required but not found, and Homebrew is not installed.\n"
                "Install Homebrew first: https://brew.sh  then run: brew install cmake"
            )
        subprocess.run(["brew", "install", "cmake"], check=True)

    elif os_name == "Linux":
        # Try apt-get (Debian/Ubuntu), fall back to yum/dnf (RHEL/Fedora)
        if shutil.which("apt-get"):
            subprocess.run(
                ["sudo", "apt-get", "update", "-y"], check=True,
            )
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "cmake"], check=True,
            )
        elif shutil.which("dnf"):
            subprocess.run(["sudo", "dnf", "install", "-y", "cmake"], check=True)
        elif shutil.which("yum"):
            subprocess.run(["sudo", "yum", "install", "-y", "cmake"], check=True)
        else:
            raise RuntimeError(
                "cmake is required but not found, and no supported package manager "
                "(apt-get, dnf, yum) was detected.\n"
                "Please install cmake manually and re-run."
            )

    elif os_name == "Windows":
        if shutil.which("winget"):
            subprocess.run(
                ["winget", "install", "--id", "Kitware.CMake",
                 "-e", "--accept-source-agreements", "--accept-package-agreements"],
                check=True,
            )
        else:
            raise RuntimeError(
                "cmake is required but not found, and winget is not available.\n"
                "Install cmake from https://cmake.org/download/ or run:\n"
                "  winget install Kitware.CMake"
            )
    else:
        raise RuntimeError(f"Unsupported OS for automatic cmake install: {os_name}")

    # Verify installation succeeded
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake was installed but still not found on PATH.\n"
            "You may need to open a new terminal or add cmake to your PATH."
        )
    print("[build-from-source] cmake installed successfully.")


def _check_build_prerequisites() -> None:
    """Ensure cmake is available; install it automatically if missing."""
    if shutil.which("cmake") is None:
        _install_cmake()


def _patch_cmake_for_arch(repo_dir: Path) -> None:
    """Patch cmake/ortlib.cmake so it correctly detects ARM64 on Apple Silicon.

    The upstream cmake only checks ``CMAKE_OSX_ARCHITECTURES`` which is empty
    on a native arm64 build (it's only set for cross-compilation).  When it's
    empty, ``ORT_BINARY_PLATFORM`` stays ``"x64"`` and the cmake later looks
    for ``runtimes/osx-x64/native/libonnxruntime.dylib`` which doesn't exist
    in the arm64 NuGet layout.

    We apply two patches (belt-and-suspenders):
      1. Regex-patch ortlib.cmake to also check CMAKE_HOST_SYSTEM_PROCESSOR.
      2. Inject CMAKE_OSX_ARCHITECTURES into CMakeLists.txt as a fallback.
    """
    machine = platform.machine().lower()
    os_name = platform.system()

    # Only patch on non-x64 platforms
    if machine not in ("arm64", "aarch64"):
        return

    patched = False

    # ---- Primary: regex-patch ortlib.cmake ----
    if os_name == "Darwin":
        ortlib_cmake = repo_dir / "cmake" / "ortlib.cmake"
        if ortlib_cmake.exists():
            content = ortlib_cmake.read_text()
            # Match the pattern with any amount of whitespace:
            #   if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            #     set(ORT_BINARY_PLATFORM "arm64")
            #   endif()
            pattern = (
                r'(if\s*\(\s*CMAKE_OSX_ARCHITECTURES\s+STREQUAL\s+"arm64"\s*\)'
                r'\s+set\s*\(\s*ORT_BINARY_PLATFORM\s+"arm64"\s*\))'
                r'(\s+endif\s*\(\s*\))'
            )
            replacement = (
                r'\1'
                r'\n      elseif(NOT CMAKE_OSX_ARCHITECTURES AND '
                r'CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")'
                r'\n        set(ORT_BINARY_PLATFORM "arm64")'
                r'\2'
            )
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                ortlib_cmake.write_text(new_content)
                print("[build-from-source] Patched ortlib.cmake: "
                      "added arm64 host-CPU detection for macOS")
                patched = True
            else:
                print("  Warning: could not find expected cmake pattern "
                      "in ortlib.cmake (primary patch).")

    # ---- Fallback: always inject CMAKE_OSX_ARCHITECTURES into CMakeLists.txt ----
    if os_name == "Darwin":
        _patch_cmake_fallback(repo_dir)
        if not patched:
            print("[build-from-source] Using CMakeLists.txt fallback patch only.")


def _patch_cmake_fallback(repo_dir: Path) -> None:
    """Fallback: set CMAKE_OSX_ARCHITECTURES in CMakeLists.txt preamble.

    If ortlib.cmake doesn't match the expected pattern (upstream changed),
    we inject an explicit ``set(CMAKE_OSX_ARCHITECTURES ...)`` early in
    CMakeLists.txt before ortlib.cmake is included.
    """
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        return

    cmakelists = repo_dir / "CMakeLists.txt"
    if not cmakelists.exists():
        return

    content = cmakelists.read_text()
    inject_line = (
        '\n# [ofoundry] Force correct architecture for native build\n'
        'if(NOT CMAKE_OSX_ARCHITECTURES)\n'
        '  set(CMAKE_OSX_ARCHITECTURES "${CMAKE_HOST_SYSTEM_PROCESSOR}")\n'
        'endif()\n'
    )

    # Insert right after the project() command
    marker = "include(cmake/options.cmake)"
    if marker in content and "[ofoundry]" not in content:
        content = content.replace(marker, marker + inject_line)
        cmakelists.write_text(content)
        print("[build-from-source] Fallback patch: injected "
              "CMAKE_OSX_ARCHITECTURES into CMakeLists.txt")


def _patch_skip_examples(repo_dir: Path) -> None:
    """Patch build.py to unconditionally skip building C examples.

    The examples require ORT headers in a separate ``ort/include`` directory
    that doesn't exist when building from a clean clone.  Some upstream
    versions may not support ``--skip_examples``, so we also patch the source
    to comment out the ``build_examples`` call.
    """
    build_py = repo_dir / "build.py"
    if not build_py.exists():
        return

    content = build_py.read_text()

    # Pattern: the call to build_examples is guarded by a condition like:
    #   if not (arguments.skip_examples or arguments.android or arguments.ios):
    #       build_examples(arguments, environment)
    # We replace it with a pass-through that always skips.
    old = "build_examples(arguments, environment)"
    new = "pass  # [ofoundry] skip examples build — only wheel needed"
    if old in content and "[ofoundry]" not in content:
        content = content.replace(old, new)
        build_py.write_text(content)
        print("[build-from-source] Patched build.py: skipped examples build")


def _is_ort_genai_from_source() -> bool:
    """Check if onnxruntime-genai is already installed from a local build.

    A source build produces a wheel whose version contains '+' or 'dev',
    or whose dist-info INSTALLER is not 'pip' from PyPI.  We check the
    version string as the simplest heuristic.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import onnxruntime_genai as og; print(og.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    version = result.stdout.strip()
    # Source builds typically have 'dev' or '+local' in the version
    return "dev" in version or "+" in version or ".post" in version


def _is_olive_from_source() -> bool:
    """Check if olive is already installed from a Git source build."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import importlib.metadata; "
         "d = importlib.metadata.distribution('olive-ai'); "
         "loc = (d.read_text('direct_url.json') or ''); "
         "print(loc)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    # PEP 610: direct_url.json contains the VCS URL for git installs
    output = result.stdout.strip()
    return "github.com/microsoft/Olive" in output


def _build_ort_genai_from_source(device: str, work_dir: Path) -> None:
    """Clone, build, and install onnxruntime-genai from source.

    Build configuration is selected automatically based on the current OS
    and the requested device target, following the official documentation:
    https://onnxruntime.ai/docs/genai/howto/build-from-source.html

    On Apple Silicon (and other non-x64 platforms) the upstream cmake
    in ``ortlib.cmake`` only checks ``CMAKE_OSX_ARCHITECTURES`` to choose
    the NuGet runtime ID.  That variable is empty on native arm64 builds,
    so it defaults to ``x64`` and looks for ``osx-x64`` — which doesn't
    exist.

    We fix this by patching ``ortlib.cmake`` to also check
    ``CMAKE_HOST_SYSTEM_PROCESSOR``, so native arm64 builds correctly
    resolve to ``osx-arm64``.
    """
    print("[build-from-source] Checking prerequisites...")
    _check_build_prerequisites()

    config = _detect_build_config(device)
    os_name = config["os_name"]
    build_args = config["build_args"]
    wheel_glob = config["wheel_glob"]

    repo_dir = work_dir / "onnxruntime-genai"
    print(f"[build-from-source] Cloning onnxruntime-genai (OS={os_name}, "
          f"device={device})...")
    subprocess.run(
        ["git", "clone", "--depth=1", ORT_GENAI_REPO, str(repo_dir)],
        check=True,
    )

    # Fix arch-related cmake issues (e.g. osx-x64 → osx-arm64)
    _patch_cmake_for_arch(repo_dir)

    # Ensure examples are skipped even if --skip_examples isn't recognized
    # by this version of build.py. We patch the main block to skip
    # build_examples() unconditionally.
    _patch_skip_examples(repo_dir)

    print(f"[build-from-source] Building Python wheel: "
          f"build.py {' '.join(build_args)} ...")
    subprocess.run(
        [sys.executable, "build.py"] + build_args,
        cwd=str(repo_dir),
        check=True,
    )

    # Find and install the built wheel
    wheels = sorted(glob.glob(str(repo_dir / wheel_glob)))
    if not wheels:
        raise RuntimeError(
            f"Build succeeded but no wheel found matching '{wheel_glob}' "
            f"under {repo_dir}.\n"
            f"Check the build output above for errors."
        )
    wheel_path = wheels[-1]  # pick the latest if multiple
    print(f"[build-from-source] Installing wheel: {wheel_path}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", wheel_path],
        check=True,
    )
    print("[build-from-source] onnxruntime-genai installed from source.")


OLIVE_GIT_URL = "git+https://github.com/microsoft/Olive.git"


def _ensure_olive_installed(from_source: bool = False) -> None:
    """Install olive-ai.

    When *from_source* is True, olive is always (re)installed from the latest
    Git main branch so that it stays compatible with the freshly-built
    onnxruntime-genai wheel.  Otherwise the PyPI release is used.
    """
    if from_source:
        print("[olive] Installing olive from source (git main)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", OLIVE_GIT_URL],
            check=True,
        )
        print("[olive] olive installed from source.")
        return

    try:
        import olive  # noqa: F401
        print("[olive] olive is already installed.")
    except ImportError:
        print("[olive] Installing olive-ai...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "olive-ai"],
            check=True,
        )
        print("[olive] olive-ai installed.")


def _get_available_olive_surgeons() -> set[str] | None:
    """Query Olive for available GraphSurgeries surgeon names.

    Returns a set of lowercase surgeon names, or None if detection fails.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "from olive.passes.onnx.graph_surgeries import Surgeon; "
         "print(','.join(Surgeon.registry.keys()))"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return set(result.stdout.strip().split(","))


def _patch_olive_config_for_surgeons(config_file: Path) -> None:
    """Remove GraphSurgeries entries that reference surgeons not in the
    installed Olive version.  This avoids failures when olive-recipes is
    newer than the local Olive installation."""
    available = _get_available_olive_surgeons()
    if available is None:
        return  # can't detect, leave config as-is

    with open(config_file) as f:
        config = json.load(f)

    passes = config.get("passes", {})
    changed = False
    for pass_name, pass_cfg in list(passes.items()):
        if pass_cfg.get("type") != "GraphSurgeries":
            continue
        surgeries = pass_cfg.get("surgeries", [])
        filtered = []
        for s in surgeries:
            name = s.get("surgeon", "").lower()
            if name in available:
                filtered.append(s)
            else:
                print(f"  Warning: removing unsupported surgeon '{s.get('surgeon')}' "
                      f"from {config_file.name} (not in installed Olive)")
                changed = True
        pass_cfg["surgeries"] = filtered

    if changed:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)


def clone_recipe(recipe_subdir: str, work_dir: Path) -> Path:
    """Shallow-clone olive-recipes and checkout only the recipe directory."""
    repo_dir = work_dir / "olive-recipes"
    subprocess.run(
        ["git", "clone", "--depth=1", "--no-checkout",
         OLIVE_RECIPES_REPO, str(repo_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "sparse-checkout", "set", recipe_subdir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout"],
        check=True,
    )
    recipe_dir = repo_dir / recipe_subdir
    if not (recipe_dir / "user_script.py").exists():
        raise FileNotFoundError(
            f"Git clone succeeded but 'user_script.py' is missing from {recipe_dir}.\n"
            "Check your network connection and try again."
        )
    return recipe_dir


def _is_qwen3_vl(model_name: str) -> bool:
    """Check if the model name refers to a Qwen3-VL architecture."""
    short = model_name.lower().replace("-", "").replace("_", "")
    return "qwen3vl" in short or "qwen3" in short and "vl" in short


def _build_text_model(
    model_name: str,
    output_dir: Path,
    precision: str = "int4",
    cache_dir: str = "./models/cache_dir",
) -> None:
    """Build the text sub-model directly via onnxruntime-genai ModelBuilder.

    Uses the native Qwen3VLTextModel when available (ort-genai >= 0.13.0-dev)
    for correct interleaved MRoPE and QK normalization.
    Falls back to Qwen25VLTextModel with manual QK-norm for older versions.
    """
    import torch
    from transformers import AutoConfig
    from onnxruntime_genai.models.builder import (
        Qwen25VLTextModel,
        set_io_dtype,
        set_onnx_dtype,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir)
    # Flatten text_config attributes onto config (same as builder dispatcher)
    text_config = config.text_config
    for key in text_config:
        if not hasattr(config, key):
            setattr(config, key, getattr(text_config, key))

    extra_options = {"exclude_embeds": True, "filename": "text.onnx"}
    io_dtype = set_io_dtype(precision, "cpu", extra_options)
    onnx_dtype = set_onnx_dtype(precision, extra_options)

    use_qwen3 = _is_qwen3_vl(model_name)
    if use_qwen3:
        try:
            from onnxruntime_genai.models.builder import Qwen3VLTextModel
            print("  Building text model (native Qwen3VLTextModel)...")
            with torch.no_grad():
                onnx_model = Qwen3VLTextModel(
                    config, io_dtype, onnx_dtype, "cpu", cache_dir, extra_options,
                )
                onnx_model.make_model(model_name)
                onnx_model.save_model(str(output_dir))
        except ImportError:
            print("  Warning: Qwen3VLTextModel not available, "
                  "falling back to Qwen25VLTextModel with QK-norm patch.")
            print("  For best results, rebuild onnxruntime-genai from source.")
            use_qwen3 = False

    if not use_qwen3:
        print("  Building text model (Qwen25VLTextModel)...")
        with torch.no_grad():
            onnx_model = Qwen25VLTextModel(
                config, io_dtype, onnx_dtype, "cpu", cache_dir, extra_options,
            )
            onnx_model.make_model(model_name)
            onnx_model.save_model(str(output_dir))

    extra_kwargs = {}
    onnx_model.make_genai_config(model_name, extra_kwargs, str(output_dir))
    onnx_model.save_processing(model_name, extra_kwargs, str(output_dir))
    print(f"  Text model saved to {output_dir}")


def _collect_outputs(search_root: Path, dest_dir: Path) -> None:
    """Recursively find all model artifacts and flatten them into dest_dir.

    Olive may place outputs in subdirectories (e.g. text.onnx/, embedding.onnx/).
    This flattens everything into a single directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(search_root.rglob("*")):
        if not path.is_file():
            continue
        dest = dest_dir / path.name
        # Keep the first genai_config.json found (from text ModelBuilder)
        if dest.exists():
            continue
        shutil.copy2(path, dest)


def _write_genai_config(models_dir: Path, device: str) -> None:
    """Create or patch genai_config.json with VL-specific model sections."""
    config_path = models_dir / "genai_config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "model": {"decoder": {"filename": "text.onnx"}},
            "search": {},
        }

    if device in ("gpu", "cuda"):
        provider_options = [
            {"cuda": {"enable_cuda_graph": "0",
                       "enable_skip_layer_norm_strict_mode": "1"}}
        ]
    else:
        provider_options = []

    session_options = {
        "log_id": "onnxruntime-genai",
        "provider_options": provider_options,
    }

    config.setdefault("model", {})
    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }
    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": 2,
        "tokens_per_second": 2.0,
        "patch_size": 16,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }
    config["model"]["image_token_id"] = 151655
    config["model"]["video_token_id"] = 151656
    config["model"]["vision_start_token_id"] = 151652

    # The C++ runtime (ort-genai <= 0.13.x) recognizes "qwen2_5_vl" for
    # the entire Qwen VL family (including Qwen3-VL).  The Python builder
    # may write "qwen3_vl" which the runtime does not accept.
    model_type = config["model"].get("type", "")
    if model_type in ("qwen3_vl", "qwen3vl"):
        config["model"]["type"] = "qwen2_5_vl"

    config.setdefault("search", {})
    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    processor_config = {
        "processor": {
            "name": "qwen3_vl_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage",
                               "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 540, "height": 360, "smart_resize": 1,
                    "min_pixels": 3136, "max_pixels": 12845056,
                    "patch_size": 16, "merge_size": 2,
                }}},
                {"operation": {"name": "rescale", "type": "Rescale", "attrs": {
                    "rescale_factor": 0.00392156862745098,
                }}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "qwen3_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                }}},
            ],
        }
    }
    processor_path = models_dir / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


def convert_vl_model(
    model_name: str,
    output_dir: str,
    device: str = "cpu",
    build_from_source: bool = False,
    cache_dir: str = "./models/cache_dir",
) -> None:
    """Convert a VL model to optimized ONNX using olive pipeline.

    When *build_from_source* is True the workflow becomes:
      0. Build onnxruntime-genai from source (platform-aware).
      1. Install olive-ai (if not present).
      2. Build text sub-model directly via ModelBuilder API.
      3. Clone olive-recipes and run Olive for embedding + vision.
      4. Flatten Olive outputs and merge with text model.
      5. Generate GenAI runtime configs.
    """
    recipe_subdir = find_recipe(model_name)
    if recipe_subdir is None:
        supported = ", ".join(VL_RECIPES.keys())
        raise ValueError(
            f"No olive recipe found for '{model_name}'.\n"
            f"Supported VL models: {supported}"
        )

    output_path = Path(output_dir).resolve()
    olive_device = "gpu" if device in ("cuda", "gpu") else "cpu"
    config_dir = "cuda" if olive_device == "gpu" else "cpu_and_mobile"

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # ── Step 0 (optional): Build onnxruntime-genai from source ──
        if build_from_source:
            if _is_ort_genai_from_source():
                print("[0/5] onnxruntime-genai (source build) already installed, "
                      "skipping rebuild.")
            else:
                print("[0/5] Building onnxruntime-genai from source...")
                _build_ort_genai_from_source(device, work_dir)

            if _is_olive_from_source():
                print("[0/5] olive (source build) already installed, "
                      "skipping reinstall.")
            else:
                print("[0/5] Installing olive from source (matching build)...")
                _ensure_olive_installed(from_source=True)

        assembled_dir = work_dir / "assembled"
        assembled_dir.mkdir()

        # ── Step 1: Text sub-model (direct ModelBuilder call) ──
        print(f"[1/4] Building text sub-model for {model_name}...")
        precision = "int4"
        _build_text_model(model_name, assembled_dir, precision=precision,
                          cache_dir=str(Path(cache_dir).resolve()))

        # ── Step 2: Clone olive recipe ──
        print(f"[2/4] Downloading olive recipe for {model_name}...")
        recipe_dir = clone_recipe(recipe_subdir, work_dir)

        config_dir_path = recipe_dir / config_dir
        if not config_dir_path.exists():
            raise FileNotFoundError(
                f"Config directory '{config_dir}' not found in {recipe_dir}.\n"
                f"Contents: {[p.name for p in recipe_dir.iterdir()]}"
            )

        # Install recipe dependencies (skip if already satisfied)
        req_file = recipe_dir / "requirements.txt"
        if req_file.exists():
            print("  Installing olive recipe dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                check=True,
            )

        # ── Step 3: Embedding + Vision via Olive ──
        print(f"[3/4] Running Olive for embedding + vision (config={config_dir})...")
        for config_name in ("embedding.json", "vision.json"):
            config_file = config_dir_path / config_name
            if not config_file.exists():
                print(f"  Warning: {config_name} not found, skipping.")
                continue
            # Remove surgeons not supported by the installed Olive version
            _patch_olive_config_for_surgeons(config_file)
            print(f"  Running Olive: {config_name}...")
            # Set HF cache dir so Olive downloads models to the same location
            olive_env = os.environ.copy()
            hf_cache = str(Path(cache_dir).resolve())
            olive_env["HF_HUB_CACHE"] = hf_cache
            olive_env["HF_HOME"] = hf_cache
            olive_env["TRANSFORMERS_CACHE"] = hf_cache
            result = subprocess.run(
                [sys.executable, "-m", "olive", "run",
                 "--config", str(config_file)],
                cwd=str(recipe_dir),
                capture_output=True,
                text=True,
                env=olive_env,
            )
            # Always show Olive stdout (progress info)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                # Show stderr so the user can diagnose the failure
                if result.stderr:
                    print(f"  [Olive stderr]\n{result.stderr}")
                raise RuntimeError(
                    f"Olive pass '{config_name}' failed (exit code {result.returncode}).\n"
                    "Check the error output above. Common causes:\n"
                    "  - Missing packages: pip install olive-ai onnxruntime torch transformers\n"
                    "  - onnxruntime-genai version mismatch after building from source\n"
                    "  - Insufficient disk space or memory for model export"
                )

        # Flatten Olive embedding/vision outputs into assembled_dir
        olive_models = config_dir_path / "models"
        if olive_models.exists():
            _collect_outputs(olive_models, assembled_dir)

        # ── Step 4: Generate GenAI runtime configs ──
        print("[4/4] Generating GenAI runtime configs...")
        _write_genai_config(assembled_dir, device=olive_device)

        # Verify essential files exist
        for required in ("text.onnx", "embedding.onnx", "vision.onnx"):
            if not (assembled_dir / required).exists():
                raise RuntimeError(
                    f"Missing required model file: {required}\n"
                    "Check the build logs above for errors."
                )

        # Copy to final output directory
        output_path.mkdir(parents=True, exist_ok=True)
        for item in assembled_dir.iterdir():
            dest = output_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    print(f"VL model saved to {output_path}")
