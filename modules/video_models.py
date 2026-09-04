"""Local model and runtime management for Fooocus image-to-video."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import modules.config


GIB = 1024 ** 3
H3_LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
MODEL_INDEX = "model_index.json"


@dataclass(frozen=True)
class VideoModel:
    key: str
    label: str
    repo_id: str
    folder_name: str
    download_bytes: int
    required_paths: tuple[str, ...]
    weight_paths: tuple[str, ...]
    allow_patterns: tuple[str, ...] | None = None
    experimental: bool = False

    @property
    def path(self) -> Path:
        return Path(modules.config.path_video_models) / self.folder_name


VIDEO_MODELS = {
    "wan": VideoModel(
        key="wan",
        label="Wan 2.2 TI2V 5B",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        folder_name="wan2.2-ti2v-5b",
        download_bytes=34_000_000_000,
        required_paths=(
            MODEL_INDEX,
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "vae/config.json",
        ),
        weight_paths=("transformer", "text_encoder", "vae"),
        allow_patterns=(
            MODEL_INDEX,
            "scheduler/**",
            "transformer/**",
            "text_encoder/**",
            "tokenizer/**",
            "vae/**",
        ),
    ),
    "h3": VideoModel(
        key="h3",
        label="MiniMax H3-Base FL2VA (experimental)",
        repo_id="MiniMaxAI/MiniMax-H3",
        folder_name="minimax-h3-fl2va",
        download_bytes=150_000_000_000,
        required_paths=(
            MODEL_INDEX,
            "modular_model_index.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "processor/preprocessor_config.json",
            "vae/config.json",
            "audio_vae/config.json",
            "scheduler/scheduler_config.json",
            "audio_scheduler/scheduler_config.json",
        ),
        weight_paths=("transformer", "text_encoder", "vae", "audio_vae"),
        allow_patterns=(
            MODEL_INDEX,
            "modular_model_index.json",
            "transformer/**",
            "text_encoder/**",
            "tokenizer/**",
            "processor/**",
            "vae/**",
            "audio_vae/**",
            "scheduler/**",
            "audio_scheduler/**",
        ),
        experimental=True,
    ),
}


def get_model(model_key: str) -> VideoModel:
    try:
        return VIDEO_MODELS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown video model: {model_key}") from exc


def model_components_present(model_key: str) -> bool:
    model = get_model(model_key)
    if not all((model.path / relative_path).is_file() for relative_path in model.required_paths):
        return False
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    return all(
        any(
            path.is_file() and path.name.endswith(weight_suffixes)
            for path in (model.path / relative_path).rglob("*")
        )
        for relative_path in model.weight_paths
    )


def model_is_ready(model_key: str) -> bool:
    model = get_model(model_key)
    return model_components_present(model_key) and (model.path / ".fooocus-complete").is_file()


def model_status(model_key: str) -> str:
    model = get_model(model_key)
    if model_is_ready(model_key):
        return f"Ready: {model.path}"
    size_gb = model.download_bytes / 1_000_000_000
    return f"Not installed · about {size_gb:.0f} GB · {model.path}"


def runtime_python() -> Path:
    root = Path(modules.config.path_video_runtime)
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _video_requirements() -> Path:
    return Path(__file__).resolve().parents[1] / "requirements_video.txt"


def _requirements_hash() -> str:
    return hashlib.sha256(_video_requirements().read_bytes()).hexdigest()


def _use_system_ca_bundle() -> None:
    """Use the OS trust store when certifi lacks a local proxy/company CA."""
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
    ):
        if Path(candidate).is_file():
            os.environ["SSL_CERT_FILE"] = candidate
            os.environ["REQUESTS_CA_BUNDLE"] = candidate
            return


def runtime_is_ready() -> bool:
    python = runtime_python()
    if not python.is_file():
        return False
    marker = Path(modules.config.path_video_runtime) / ".fooocus-video-runtime"
    if not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        return state.get("requirements_hash") == _requirements_hash()
    except (OSError, ValueError):
        return False


def _parse_video_requirements() -> tuple[list[str], list[str]]:
    """Split torch wheels (PyTorch index) from the remaining PyPI packages."""
    torch_packages: list[str] = []
    other_packages: list[str] = []
    for raw in _video_requirements().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        name = line.split("==", 1)[0].split("[", 1)[0].strip().lower()
        if name in {"torch", "torchvision", "torchaudio", "torchao"}:
            torch_packages.append(line)
        else:
            other_packages.append(line)
    return torch_packages, other_packages


def _run_pip(command: list[str], report: Callable[[str], None]) -> None:
    """Run a pip command, stream progress, and raise with useful output on failure."""
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError("Could not read output from the video environment installer.")
    last_lines: list[str] = []
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        last_lines.append(line)
        if len(last_lines) > 40:
            last_lines = last_lines[-40:]
        report(line)
    return_code = process.wait()
    if return_code != 0:
        detail = "\n".join(last_lines[-12:]) or "(no pip output)"
        raise RuntimeError(
            f"Video environment setup failed with exit code {return_code}.\n{detail}"
        )


def _pip_base(python: str, index_url: str, trusted_host: str | None = None) -> list[str]:
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--timeout",
        "120",
        "-i",
        index_url,
    ]
    if trusted_host:
        command.extend(["--trusted-host", trusted_host])
    return command


def _run_import_probe(python: str, script: str, failure_label: str) -> str:
    probe = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or f"{failure_label} failed.").strip()
        raise RuntimeError(f"{failure_label}:\n{detail}")
    return probe.stdout.strip()


def _try_pypi_indexes(
    python: str,
    packages: list[str],
    report: Callable[[str], None],
    label: str,
    *,
    extra_args: list[str] | None = None,
) -> None:
    """Install packages trying the primary index, then a mirror fallback."""
    from urllib.parse import urlparse

    pypi_index = os.environ.get("FOOOCUS_VIDEO_PYPI_INDEX", "https://pypi.org/simple")
    mirror_index = os.environ.get(
        "FOOOCUS_VIDEO_PYPI_MIRROR",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
    )
    # Prefer a working mirror first when the official index is flaky (common SSL resets).
    indexes = [mirror_index]
    if pypi_index not in indexes:
        indexes.append(pypi_index)

    errors: list[str] = []
    for index_url in indexes:
        host = urlparse(index_url).hostname
        trusted = host if "pypi.org" not in index_url else None
        report(f"{label} using {index_url} …")
        try:
            command = _pip_base(python, index_url, trusted)
            if extra_args:
                command.extend(extra_args)
            command.extend(packages)
            _run_pip(command, report)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
            report(f"{label} failed on {index_url}; trying next source …")
    raise RuntimeError("\n\n".join(errors) if errors else f"{label} failed.")


def setup_runtime(progress: Callable[[str], None] | None = None) -> Path:
    """Create the isolated environment and install pinned video dependencies."""
    report = progress or (lambda _message: None)
    runtime_dir = Path(modules.config.path_video_runtime)
    requirements = _video_requirements()
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    if runtime_is_ready():
        report("Video environment is already up to date.")
        return runtime_python()

    if not runtime_python().exists():
        report("Creating isolated video environment …")
        venv.EnvBuilder(with_pip=True, clear=False).create(runtime_dir)

    python = str(runtime_python())
    torch_index = os.environ.get(
        "FOOOCUS_VIDEO_TORCH_INDEX",
        "https://download.pytorch.org/whl/cu128",
    )
    torch_packages, other_packages = _parse_video_requirements()

    report("Ensuring pip tooling is available …")
    try:
        _try_pypi_indexes(python, ["pip", "setuptools", "wheel"], report, "Upgrading pip")
    except RuntimeError as exc:
        report(
            "Pip upgrade skipped because package indexes are unreachable. "
            f"Continuing with the existing pip.\n{exc}"
        )

    if torch_packages:
        from urllib.parse import urlparse

        mirror_index = os.environ.get(
            "FOOOCUS_VIDEO_PYPI_MIRROR",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        )
        mirror_host = urlparse(mirror_index).hostname or "pypi.tuna.tsinghua.edu.cn"
        # Satisfy ordinary Python dependencies from the working PyPI mirror first.
        # The subsequent Torch install can then resolve only Torch's CUDA companion
        # wheels from the official PyTorch index, without reaching pythonhosted.org.
        report("Installing Torch Python dependencies from mirror …")
        _run_pip(
            [
                python,
                "-m",
                "pip",
                "install",
                "--retries",
                "10",
                "--timeout",
                "120",
                "-i",
                mirror_index,
                "--trusted-host",
                mirror_host,
                "filelock",
                "typing-extensions>=4.10.0",
                "sympy",
                "networkx",
                "jinja2",
                "fsspec",
                "pillow",
                "numpy",
                "cuda-bindings==12.9.4",
            ],
            report,
        )
        report("Installing Torch and matching CUDA companion wheels …")
        _run_pip(
            [
                python,
                "-m",
                "pip",
                "install",
                "--retries",
                "10",
                "--timeout",
                "120",
                "--index-url",
                torch_index,
                *torch_packages,
            ],
            report,
        )
        torch_version = _run_import_probe(
            python,
            "import torch, torchvision, torchaudio; print(torch.__version__)",
            "Torch packages installed but failed to import",
        )
        report(f"Torch ready: {torch_version}")

    if other_packages:
        # Keep already-installed Torch/CUDA wheels; only pull missing helper deps.
        # Install the pinned packages without deps first so PyPI cannot replace the
        # CUDA companions that came from the PyTorch index.
        _try_pypi_indexes(
            python,
            other_packages,
            report,
            "Installing Diffusers and video helper packages (no deps)",
            extra_args=["--no-deps", "--upgrade-strategy", "only-if-needed"],
        )
        helper_deps = [
            "importlib-metadata",
            "filelock",
            "requests",
            "tqdm",
            "regex",
            "pyyaml",
            "packaging",
            "tokenizers>=0.22.0,<0.24.0",
            "hf-xet",
            "httpx",
            "typer",
            "rich",
        ]
        _try_pypi_indexes(
            python,
            helper_deps,
            report,
            "Installing Diffusers/Transformers helper dependencies",
            extra_args=["--upgrade-strategy", "only-if-needed"],
        )
        if torch_packages:
            torch_version = _run_import_probe(
                python,
                (
                    "import torch, diffusers, transformers; "
                    "from torchao.quantization import Int8WeightOnlyConfig; "
                    "Int8WeightOnlyConfig(version=2); print(torch.__version__)"
                ),
                "Video packages installed but failed to import",
            )
            report(f"Video stack ready (Torch {torch_version})")

    marker = runtime_dir / ".fooocus-video-runtime"
    marker.write_text(
        json.dumps(
            {
                "requirements": str(requirements),
                "requirements_hash": _requirements_hash(),
                "created_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    report("Video environment is ready.")
    return runtime_python()


def _download_model_process(model_key: str, status_file: str) -> None:
    model = get_model(model_key)
    status_path = Path(status_file)
    try:
        _use_system_ca_bundle()
        from huggingface_hub import snapshot_download

        status_path.write_text("Connecting to Hugging Face …", encoding="utf-8")
        model.path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model.repo_id,
            local_dir=str(model.path),
            allow_patterns=list(model.allow_patterns) if model.allow_patterns else None,
        )
        if not model_components_present(model_key):
            raise RuntimeError("Download completed but required model components are missing.")
        (model.path / ".fooocus-complete").write_text(model.repo_id, encoding="utf-8")
        status_path.write_text("complete", encoding="utf-8")
    except Exception as exc:
        status_path.write_text(f"error: {exc}", encoding="utf-8")
        raise


class DownloadManager:
    """Owns the current resumable model download so the UI can cancel it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: multiprocessing.Process | None = None
        self._status_file: Path | None = None
        self._model_key: str | None = None

    def start(self, model_key: str) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise RuntimeError("Another video model download is already running.")
            model = get_model(model_key)
            free_bytes = shutil.disk_usage(model.path.parent).free
            remaining = max(0, model.download_bytes - directory_size(model.path))
            if free_bytes < remaining + 2 * GIB:
                raise RuntimeError(
                    f"Not enough free disk space. Need about {remaining / GIB:.1f} GiB "
                    f"plus 2 GiB working space."
                )
            status_dir = Path(modules.config.path_video_runtime)
            status_dir.mkdir(parents=True, exist_ok=True)
            self._status_file = status_dir / "download-status.txt"
            self._status_file.write_text(f"Starting {model.label} download …", encoding="utf-8")
            self._model_key = model_key
            self._process = multiprocessing.get_context("spawn").Process(
                target=_download_model_process,
                args=(model_key, str(self._status_file)),
                daemon=True,
            )
            self._process.start()

    def cancel(self) -> bool:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                return False
            self._process.terminate()
            self._process.join(timeout=5)
            if self._status_file:
                self._status_file.write_text(
                    "Cancelled. Run Download again to resume.", encoding="utf-8"
                )
            return True

    def status(self) -> tuple[bool, str]:
        with self._lock:
            running = self._process is not None and self._process.is_alive()
            if self._status_file and self._status_file.exists():
                message = self._status_file.read_text(encoding="utf-8")
            else:
                message = "Idle"
            if running and self._model_key:
                model = get_model(self._model_key)
                downloaded = directory_size(model.path)
                percent = min(99, int(100 * downloaded / model.download_bytes))
                message = (
                    f"{message}\nDownloaded about {downloaded / GIB:.1f} of "
                    f"{model.download_bytes / GIB:.1f} GiB ({percent}%)."
                )
            if self._process is not None and not running and self._process.exitcode:
                if not message.startswith("error:") and not message.startswith("Cancelled"):
                    message = f"Download stopped with exit code {self._process.exitcode}."
            return running, message


DOWNLOAD_MANAGER = DownloadManager()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            try:
                total += (Path(root) / file_name).stat().st_size
            except OSError:
                pass
    return total


def hardware_info() -> dict[str, float | str | bool]:
    total_ram = 0
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
    except Exception:
        pass

    vram = 0
    cuda = False
    device_name = "CPU"
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            vram = properties.total_memory
            device_name = properties.name
    except Exception:
        pass

    free_disk = shutil.disk_usage(modules.config.path_video_models).free
    return {
        "cuda": cuda,
        "device_name": device_name,
        "vram_gb": round(vram / GIB, 1),
        "ram_gb": round(total_ram / GIB, 1),
        "free_disk_gb": round(free_disk / GIB, 1),
    }


def resolve_hardware_profile(requested: str = "Auto") -> str:
    if requested != "Auto":
        return requested
    vram = float(hardware_info()["vram_gb"])
    if vram < 10:
        return "8 GB"
    if vram < 14:
        return "12 GB"
    return "16 GB+"


def preflight(model_key: str, profile: str = "Auto") -> tuple[bool, str]:
    model = get_model(model_key)
    info = hardware_info()
    selected = resolve_hardware_profile(profile)
    if not info["cuda"]:
        return False, "A CUDA-capable NVIDIA GPU is required for local video generation."

    if model_key == "h3":
        if not h3_authorized():
            return False, "Confirm your MiniMax H3 authorization before downloading or running H3."
        warnings = []
        if float(info["vram_gb"]) < 12:
            warnings.append("H3 is unsupported below 12 GB VRAM")
        if float(info["ram_gb"]) < 75:
            warnings.append("H3 offload normally needs about 75 GB system RAM")
        remaining_gib = max(0, model.download_bytes - directory_size(model.path)) / GIB
        if float(info["free_disk_gb"]) < remaining_gib:
            warnings.append(f"the official FL2VA files need about {remaining_gib:.0f} GiB more free disk")
        if warnings:
            return False, "; ".join(warnings) + "."
        return True, "H3 experimental preflight passed."

    if selected == "8 GB":
        return True, "8 GB mode uses reduced frames/resolution and maximum CPU offload; generation is slow."
    return True, f"Wan profile: {selected}."


def _h3_marker() -> Path:
    return Path(modules.config.path_video_models) / ".minimax-h3-authorization"


def set_h3_authorized(authorized: bool) -> None:
    marker = _h3_marker()
    if authorized:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"acknowledged": True, "license": H3_LICENSE_URL}),
            encoding="utf-8",
        )
    elif marker.exists():
        marker.unlink()


def h3_authorized() -> bool:
    return _h3_marker().is_file()


def model_status_summary() -> str:
    info = hardware_info()
    lines = [
        f"GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)",
        f"System RAM: {info['ram_gb']} GB · Free disk: {info['free_disk_gb']} GB",
        f"Wan: {model_status('wan')}",
        f"MiniMax H3: {model_status('h3')}",
    ]
    return "\n".join(lines)
