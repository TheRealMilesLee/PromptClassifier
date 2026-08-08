from __future__ import annotations

import unittest

from trendmodel.tf_runtime import inspect_tensorflow_runtime


class _Device:
    def __init__(self, name: str) -> None:
        self.name = name


class _Experimental:
    @staticmethod
    def get_device_details(device: _Device) -> dict[str, object]:
        return {"device_name": device.name, "compute_capability": (8, 9)}


class _Config:
    experimental = _Experimental()

    def __init__(self, gpu: bool) -> None:
        self._physical = [_Device("/physical_device:CPU:0")]
        self._gpu = [_Device("/physical_device:GPU:0")] if gpu else []

    def list_physical_devices(self, kind: str | None = None) -> list[_Device]:
        if kind == "GPU":
            return self._gpu
        return self._physical + self._gpu


class _Sysconfig:
    @staticmethod
    def get_build_info() -> dict[str, object]:
        return {"is_cuda_build": True, "cuda_version": "12.3"}


class _CudaTensorFlow:
    __version__ = "2.16.2"
    config = _Config(gpu=True)
    sysconfig = _Sysconfig()

    class test:
        @staticmethod
        def is_built_with_cuda() -> bool:
            return True


class _CpuTensorFlow:
    __version__ = "2.16.2"
    config = _Config(gpu=False)
    sysconfig = _Sysconfig()

    class test:
        @staticmethod
        def is_built_with_cuda() -> bool:
            return True


class TensorFlowRuntimeTest(unittest.TestCase):
    def test_cuda_runtime_is_identified(self) -> None:
        runtime = inspect_tensorflow_runtime(_CudaTensorFlow)
        self.assertEqual(runtime["accelerator"], "cuda")
        self.assertTrue(runtime["cuda_build"])
        self.assertEqual(runtime["gpu_details"]["/physical_device:GPU:0"]["compute_capability"], [8, 9])

    def test_no_gpu_is_identified_as_cpu(self) -> None:
        runtime = inspect_tensorflow_runtime(_CpuTensorFlow)
        self.assertEqual(runtime["accelerator"], "cpu")
        self.assertEqual(runtime["gpu_devices"], [])


if __name__ == "__main__":
    unittest.main()
