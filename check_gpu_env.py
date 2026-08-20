#!/usr/bin/env python3
"""
Check whether the local NVIDIA GPU, CUDA/cuDNN runtime, and PyTorch install match
well enough to run CUDA workloads.

Usage:
  python check_gpu_env.py
  python check_gpu_env.py --json
  python check_gpu_env.py --strict
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    level: str
    name: str
    message: str
    detail: str | None = None


@dataclass
class Report:
    facts: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    def ok(self, name: str, message: str, detail: str | None = None) -> None:
        self.checks.append(CheckResult("OK", name, message, detail))

    def warn(self, name: str, message: str, detail: str | None = None) -> None:
        self.checks.append(CheckResult("WARN", name, message, detail))

    def fail(self, name: str, message: str, detail: str | None = None) -> None:
        self.checks.append(CheckResult("FAIL", name, message, detail))

    def has_failures(self) -> bool:
        return any(item.level == "FAIL" for item in self.checks)

    def has_warnings(self) -> bool:
        return any(item.level == "WARN" for item in self.checks)


def run_cmd(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    path = shutil.which(cmd[0])
    if path is None:
        return {
            "found": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{cmd[0]} not found in PATH",
        }
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "found": True,
            "path": path,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "found": True,
            "path": path,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout}s",
        }


def parse_version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def version_gt(a: tuple[int, ...] | None, b: tuple[int, ...] | None) -> bool:
    if a is None or b is None:
        return False
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def collect_system(report: Report) -> None:
    report.facts["system"] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "CUDA_PATH",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "PATH",
            )
            if os.environ.get(key)
        },
    }
    report.ok("Python", f"Python executable: {sys.executable}")


def collect_nvidia(report: Report) -> None:
    nvidia_smi = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    report.facts["nvidia_smi"] = nvidia_smi

    if not nvidia_smi["found"]:
        report.warn(
            "nvidia-smi",
            "nvidia-smi was not found.",
            "This is expected on machines without NVIDIA drivers, but CUDA PyTorch cannot use an NVIDIA GPU without the driver stack.",
        )
        return

    if nvidia_smi["returncode"] != 0:
        fallback = run_cmd(["nvidia-smi"])
        report.facts["nvidia_smi_fallback"] = fallback
        if fallback["returncode"] != 0:
            report.fail("nvidia-smi", "nvidia-smi exists but failed.", nvidia_smi["stderr"])
            return

        output = fallback["stdout"]
        driver_version = first_match(r"Driver Version:\s*([0-9.]+)", output)
        cuda_version = first_match(r"CUDA Version:\s*([0-9.]+)", output)
        gpu_rows = re.findall(r"\|\s+\d+\s+(.+?)\s{2,}On\s+\|", output)
        if not gpu_rows:
            gpu_rows = re.findall(r"\|\s+\d+\s+(.+?)\s{2,}(?:Off|On)\s+\|", output)

        report.facts["nvidia_driver_version"] = driver_version
        report.facts["driver_supported_cuda"] = cuda_version
        report.facts["gpus_from_nvidia_smi"] = gpu_rows
        report.warn(
            "nvidia-smi query",
            "Structured nvidia-smi query failed, but plain nvidia-smi works.",
            nvidia_smi["stderr"] or "Some nvidia-smi versions do not support every --query-gpu field.",
        )
        if gpu_rows:
            report.ok("NVIDIA GPU", f"nvidia-smi sees {len(gpu_rows)} GPU(s).", "\n".join(gpu_rows))
        else:
            report.ok("NVIDIA GPU", "plain nvidia-smi works; GPU table parsing was skipped.")
        return

    rows = [line.strip() for line in nvidia_smi["stdout"].splitlines() if line.strip()]
    report.facts["gpus_from_nvidia_smi"] = rows
    if not rows:
        report.fail("NVIDIA GPU", "nvidia-smi returned no GPUs.")
        return

    report.ok("NVIDIA GPU", f"nvidia-smi sees {len(rows)} GPU(s).", "\n".join(rows))

    driver_version = rows[0].split(",")[1].strip() if len(rows[0].split(",")) >= 2 else None
    report.facts["nvidia_driver_version"] = driver_version

    plain = run_cmd(["nvidia-smi"])
    report.facts["nvidia_smi_plain"] = plain
    if plain["returncode"] == 0:
        report.facts["driver_supported_cuda"] = first_match(r"CUDA Version:\s*([0-9.]+)", plain["stdout"])


def collect_cuda_toolkit(report: Report) -> None:
    nvcc = run_cmd(["nvcc", "--version"])
    report.facts["nvcc"] = nvcc

    if not nvcc["found"]:
        report.warn(
            "CUDA toolkit",
            "nvcc was not found in PATH.",
            "PyTorch wheels can run without local nvcc, but compiling custom CUDA extensions will need a CUDA toolkit.",
        )
        return

    if nvcc["returncode"] != 0:
        report.warn("CUDA toolkit", "nvcc exists but failed.", nvcc["stderr"])
        return

    release = first_match(r"release\s+([0-9.]+)", nvcc["stdout"])
    report.facts["nvcc_cuda_release"] = release
    report.ok("CUDA toolkit", f"nvcc is available; CUDA toolkit release: {release or 'unknown'}.")


def collect_cudnn_library(report: Report) -> None:
    found = {
        "cudnn": ctypes.util.find_library("cudnn"),
        "cudnn_ops": ctypes.util.find_library("cudnn_ops"),
        "cudnn_cnn": ctypes.util.find_library("cudnn_cnn"),
    }
    report.facts["cudnn_libraries"] = found
    if any(found.values()):
        report.ok("cuDNN library", f"ctypes can locate cuDNN-related library entries: {found}")
    else:
        report.warn(
            "cuDNN library",
            "ctypes could not locate system cuDNN libraries.",
            "This can be fine for PyTorch wheels that bundle cuDNN. The PyTorch cuDNN check below is more important.",
        )


def torch_smoke_test(report: Report, torch: Any) -> None:
    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        report.fail("PyTorch CUDA", f"torch.cuda.device_count() failed: {exc!r}")
        return

    if device_count <= 0:
        report.fail("PyTorch CUDA", "torch.cuda.is_available() is true, but device_count is 0.")
        return

    smoke_results = []
    for idx in range(device_count):
        try:
            device = torch.device(f"cuda:{idx}")
            torch.cuda.set_device(device)
            props = torch.cuda.get_device_properties(device)
            a = torch.randn((1024, 1024), device=device)
            b = torch.randn((1024, 1024), device=device)
            start = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if not torch.isfinite(c).all().item():
                raise RuntimeError("matmul produced non-finite values")

            conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
            x = torch.randn((4, 3, 128, 128), device=device)
            y = conv(x)
            loss = y.square().mean()
            loss.backward()
            torch.cuda.synchronize(device)

            smoke_results.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "matmul_ms": round(elapsed_ms, 2),
                }
            )
        except Exception as exc:
            report.fail("GPU smoke test", f"CUDA smoke test failed on cuda:{idx}.", repr(exc))
            return

    report.facts["torch_gpu_smoke_test"] = smoke_results
    report.ok("GPU smoke test", f"Tensor matmul and cuDNN-style Conv2d passed on {device_count} GPU(s).")


def collect_torch(report: Report) -> None:
    try:
        import torch
    except Exception as exc:
        report.fail("PyTorch", f"Failed to import torch: {exc!r}")
        return

    torch_info: dict[str, Any] = {
        "version": getattr(torch, "__version__", None),
        "module_file": getattr(torch, "__file__", None),
        "compiled_cuda": getattr(torch.version, "cuda", None),
        "compiled_hip": getattr(torch.version, "hip", None),
        "cuda_is_available": None,
        "cuda_device_count": None,
        "cudnn_enabled": bool(getattr(torch.backends.cudnn, "enabled", False)),
        "cudnn_version": None,
        "arch_list": None,
    }

    try:
        torch_info["cuda_is_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        report.fail("PyTorch CUDA", f"PyTorch CUDA query failed: {exc!r}")

    try:
        torch_info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:
        torch_info["cudnn_version_error"] = repr(exc)

    try:
        if hasattr(torch.cuda, "get_arch_list"):
            torch_info["arch_list"] = torch.cuda.get_arch_list()
    except Exception as exc:
        torch_info["arch_list_error"] = repr(exc)

    report.facts["torch"] = torch_info
    report.ok("PyTorch", f"Imported torch {torch_info['version']} from {torch_info['module_file']}.")

    compiled_cuda = torch_info["compiled_cuda"]
    if not compiled_cuda:
        report.fail(
            "PyTorch CUDA build",
            "This PyTorch build is CPU-only or not compiled with CUDA.",
            "Install a CUDA-enabled PyTorch build if you expect NVIDIA GPU acceleration.",
        )
        return

    report.ok("PyTorch CUDA build", f"PyTorch was compiled with CUDA {compiled_cuda}.")

    driver_supported_cuda = parse_version_tuple(report.facts.get("driver_supported_cuda"))
    torch_cuda = parse_version_tuple(compiled_cuda)
    if driver_supported_cuda and torch_cuda:
        if version_gt(torch_cuda, driver_supported_cuda):
            report.warn(
                "Driver/runtime compatibility",
                f"PyTorch CUDA {compiled_cuda} is newer than the CUDA version shown by nvidia-smi ({report.facts.get('driver_supported_cuda')}).",
                "This can still work through CUDA minor-version compatibility or container compatibility libraries. The GPU smoke test below is the deciding runtime check.",
            )
        else:
            report.ok(
                "Driver/runtime compatibility",
                f"NVIDIA driver advertises CUDA {report.facts.get('driver_supported_cuda')}, which is compatible with PyTorch CUDA {compiled_cuda}.",
            )

    nvcc_cuda = parse_version_tuple(report.facts.get("nvcc_cuda_release"))
    if nvcc_cuda and torch_cuda:
        if nvcc_cuda[:2] != torch_cuda[:2]:
            report.warn(
                "Toolkit/PyTorch CUDA version",
                f"nvcc CUDA {report.facts.get('nvcc_cuda_release')} differs from PyTorch CUDA {compiled_cuda}.",
                "Runtime can still work. For custom CUDA extensions, using matching major/minor versions avoids build and ABI issues.",
            )
        else:
            report.ok("Toolkit/PyTorch CUDA version", "nvcc CUDA and PyTorch CUDA major/minor versions match.")

    if torch_info["cuda_is_available"]:
        report.ok("PyTorch CUDA availability", f"torch.cuda sees {torch_info['cuda_device_count']} CUDA device(s).")
    else:
        report.fail(
            "PyTorch CUDA availability",
            "torch.cuda.is_available() is false.",
            "Check NVIDIA driver, CUDA_VISIBLE_DEVICES, container GPU passthrough, and the PyTorch CUDA build.",
        )
        return

    if torch_info["cudnn_version"]:
        report.ok("PyTorch cuDNN", f"PyTorch reports cuDNN version {torch_info['cudnn_version']}.")
    else:
        report.warn("PyTorch cuDNN", "PyTorch did not report a cuDNN version.")

    torch_smoke_test(report, torch)


def print_text_report(report: Report) -> None:
    print("\n=== GPU / CUDA / cuDNN / PyTorch Environment Check ===\n")
    print("System:")
    sys_info = report.facts.get("system", {})
    for key in ("python_executable", "platform", "machine", "cwd"):
        if key in sys_info:
            print(f"  {key}: {sys_info[key]}")
    if sys_info.get("env"):
        print("  selected env vars:")
        for key, value in sys_info["env"].items():
            if key in {"PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"} and value:
                parts = value.split(os.pathsep)
                value = os.pathsep.join(parts[:6]) + (" ..." if len(parts) > 6 else "")
            print(f"    {key}={value}")

    print("\nChecks:")
    for item in report.checks:
        print(f"  [{item.level:<4}] {item.name}: {item.message}")
        if item.detail:
            detail = textwrap.indent(item.detail.strip(), "         ")
            print(detail)

    print("\nSummary:")
    if report.has_failures():
        print("  FAIL: CUDA/PyTorch GPU workload is not ready. Fix the FAIL items above first.")
    elif report.has_warnings():
        print("  WARN: Basic GPU workload passed or no hard failure was found, but review WARN items.")
    else:
        print("  OK: GPU, CUDA runtime, cuDNN through PyTorch, and PyTorch smoke tests look healthy.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPU/CUDA/cuDNN/PyTorch compatibility.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when warnings are present, not only failures",
    )
    args = parser.parse_args()

    report = Report()
    collect_system(report)
    collect_nvidia(report)
    collect_cuda_toolkit(report)
    collect_cudnn_library(report)
    collect_torch(report)

    if args.json:
        print(
            json.dumps(
                {
                    "facts": report.facts,
                    "checks": [item.__dict__ for item in report.checks],
                    "summary": {
                        "has_failures": report.has_failures(),
                        "has_warnings": report.has_warnings(),
                    },
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print_text_report(report)

    if report.has_failures():
        return 2
    if args.strict and report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
