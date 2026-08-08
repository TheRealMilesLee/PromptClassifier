"""TensorFlow runtime and accelerator inspection helpers."""

from __future__ import annotations

import platform
from typing import Any


def inspect_tensorflow_runtime(tf: Any) -> dict[str, Any]:
    """Return JSON-safe TensorFlow/device metadata for a training artifact.

    Apple Silicon exposes Metal through a GPU device, while the same generic
    device query is used for CUDA.  The accelerator label therefore needs to
    inspect both the TensorFlow build and the host platform instead of merely
    checking whether a GPU exists.
    """

    config = tf.config
    physical_devices = list(config.list_physical_devices())
    gpu_devices = list(config.list_physical_devices("GPU"))

    build_info: dict[str, Any] = {}
    sysconfig = getattr(tf, "sysconfig", None)
    get_build_info = getattr(sysconfig, "get_build_info", None)
    if callable(get_build_info):
        try:
            raw_build_info = get_build_info()
            if isinstance(raw_build_info, dict):
                build_info = _json_safe(raw_build_info)
        except Exception:  # pragma: no cover - depends on TensorFlow build
            build_info = {}

    cuda_build = _is_cuda_build(tf, build_info)
    host_system = platform.system()
    if not gpu_devices:
        accelerator = "cpu"
    elif cuda_build:
        accelerator = "cuda"
    elif host_system == "Darwin":
        accelerator = "metal"
    else:
        # Keep an explicit fallback for other TensorFlow GPU plugins instead
        # of mislabelling them as CUDA or Metal.
        accelerator = "gpu"

    device_details: dict[str, Any] = {}
    experimental = getattr(config, "experimental", None)
    get_device_details = getattr(experimental, "get_device_details", None)
    if callable(get_device_details):
        for device in gpu_devices:
            device_name = str(getattr(device, "name", device))
            try:
                device_details[device_name] = _json_safe(get_device_details(device))
            except Exception:  # pragma: no cover - depends on installed plugin
                device_details[device_name] = {}

    return {
        "platform": host_system,
        "python_version": platform.python_version(),
        "tensorflow_version": str(getattr(tf, "__version__", "unknown")),
        "physical_devices": [str(getattr(device, "name", device)) for device in physical_devices],
        "gpu_devices": [str(getattr(device, "name", device)) for device in gpu_devices],
        "accelerator": accelerator,
        "cuda_build": cuda_build,
        "build_info": build_info,
        "gpu_details": device_details,
    }


def _is_cuda_build(tf: Any, build_info: dict[str, Any]) -> bool:
    test_api = getattr(tf, "test", None)
    is_built_with_cuda = getattr(test_api, "is_built_with_cuda", None)
    if callable(is_built_with_cuda):
        try:
            return bool(is_built_with_cuda())
        except Exception:  # pragma: no cover - depends on TensorFlow build
            pass
    return bool(build_info.get("is_cuda_build", False))


def _json_safe(value: Any) -> Any:
    """Convert common TensorFlow/numpy values into JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # pragma: no cover - defensive conversion fallback
            pass
    return str(value)
