import time
from typing import Callable, Iterable

import numpy as np
import torch


def naive_loop(x: Iterable) -> None:
    sum_ = 0.0
    for i in x:
        sum_ += i


def numpy_vector_op(x: np.array) -> None:
    sum_ = np.sum(x)


def torch_cpu_or_gpu(x: torch.Tensor) -> None:
    sum_ = torch.sum(x)


def estimate_flops(
    f: Callable[[int], float], n: int, lib: str = "numpy", iterations: int = 10
) -> float:
    times_ns = []
    if lib == "native":
        x = [1.0] * n
    elif lib == "numpy":
        x = np.ones(n, dtype=np.float64)
    elif lib == "torch_cpu":
        x = torch.ones(n, dtype=torch.float64)
    elif lib == "torch_gpu":
        x = torch.randn(n, dtype=torch.float64).to(torch.device("cuda:0"))
    else:
        raise RuntimeError(f"Invalid library {lib} requested!")

    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        f(x)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    avg_time_ns = np.average(np.array(times_ns))

    return n / (avg_time_ns / int(1e9))


if __name__ == "__main__":
    n = 1e6
    naivest_loop_flops = estimate_flops(naive_loop, int(n), "native")
    print(f"Python list, naive for loop: {naivest_loop_flops:.2E}")

    naive_loop_flops = estimate_flops(naive_loop, int(n), "native")
    print(f"NumPy array, naive for loop: {naive_loop_flops:.2E}")

    n = 1e8
    numpy_flops = estimate_flops(numpy_vector_op, int(n), "numpy")
    print(f"NumPy array, vectorized sum: {numpy_flops:.2E}")

    torch_cpu_flops = estimate_flops(torch_cpu_or_gpu, int(n), "torch_cpu")
    print(f"Torch Tensor (CPU), vectorized sum: {numpy_flops:.2E}")

    torch_gpu_flops = estimate_flops(torch_cpu_or_gpu, int(n), "torch_gpu")
    print(f"Torch Tensor (GPU), vectorized sum: {numpy_flops:.2E}")
