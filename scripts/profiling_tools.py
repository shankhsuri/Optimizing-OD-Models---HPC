import time
import torch
import GPUtil

class GPUProfiler:
    def __init__(self):
        self._start_time = None
        self._gpu_utilization = None

    def __enter__(self):
        self._start_time = time.time()
        self._gpu_utilization = GPUtil.getGPUs()[0].load
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self._start_time
        gpu_utilization = GPUtil.getGPUs()[0].load
        print(f"Elapsed time: {elapsed_time} s")
        print(f"GPU utilization increase: {gpu_utilization - self._gpu_utilization}")

    @staticmethod
    def is_cuda_available():
        return torch.cuda.is_available()

    @staticmethod
    def get_device():
        return torch.device("cuda" if GPUProfiler.is_cuda_available() else "cpu")
