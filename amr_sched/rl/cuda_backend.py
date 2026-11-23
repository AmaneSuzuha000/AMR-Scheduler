"""CUDA / Multi-Core Compute Backend for RL Policy Training and Batched Inference.

Automatically leverages NVIDIA GPU devices (e.g., RTX 5060 via CUDA 12.8/13.0 PyTorch wheels)
when available and degrades gracefully to multi-threaded CPU tensors when running headless.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Optional

import numpy as np

# Guard against CPython 3.13+ Windows WMI COM deadlock (`_wmi_query` on concurrent torch/scipy imports)
if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def get_rl_device(prefer_gpu: bool = True, *, prefer_cuda: Optional[bool] = None) -> Any:
    """Return best available PyTorch device (`cuda:0` when GPU requested and CUDA works, else `cpu`)."""
    if torch is None:
        return None
    use_gpu = prefer_gpu if prefer_cuda is None else bool(prefer_cuda)
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def cuda_runtime_info() -> dict[str, Any]:
    """Return deterministic hardware/driver telemetry for logging and benchmark provenance."""
    if torch is None:
        return {"torch_available": False, "cuda_available": False, "device_name": None}
    avail = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if avail else None
    cap = tuple(torch.cuda.get_device_capability(0)) if avail else None
    mem_gb = (
        round(float(torch.cuda.get_device_properties(0).total_memory) / (1024.0 ** 3), 2)
        if avail
        else 0.0
    )
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": avail,
        "device_name": name,
        "compute_capability": cap,
        "vram_gb": mem_gb,
    }


def move_module_to_device(module: Any, prefer_gpu: bool = True) -> tuple[Any, Any]:
    """Move a PyTorch `nn.Module` onto `cuda:0` if available and return `(module, device)`."""
    dev = get_rl_device(prefer_gpu=prefer_gpu)
    if dev is None or module is None:
        return module, dev
    return module.to(dev), dev


def batch_infer_logits(
    module: Any,
    features: np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 4096,
) -> tuple[np.ndarray, ...]:
    """Run no-grad GPU/CPU batched inference over a 2D NumPy feature matrix cleanly."""
    if torch is None or module is None:
        raise RuntimeError("PyTorch module required for batched inference")
    dev = device or get_rl_device(prefer_gpu=True)
    x = np.asarray(features, dtype=np.float32)
    n = int(x.shape[0])
    if n == 0:
        return (np.zeros((0,), dtype=np.float32),)
    was_training = bool(getattr(module, "training", False))
    module.eval()
    outs: list[list[np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, n, max(1, int(batch_size))):
            xb = torch.from_numpy(x[start : start + batch_size]).to(dev, non_blocking=(dev.type == "cuda"))
            raw = module(xb)
            parts = raw if isinstance(raw, tuple) else (raw,)
            if not outs:
                outs = [[] for _ in parts]
            for idx, tensor in enumerate(parts):
                outs[idx].append(tensor.detach().cpu().numpy().astype(np.float32))
    if was_training:
        module.train()
    return tuple(np.concatenate(chunks, axis=0) for chunks in outs)
