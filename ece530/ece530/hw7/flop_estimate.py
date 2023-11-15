#!/usr/bin/env python3
import time
from typing import Callable, Iterable

import numpy as np
import torch

N_FACT = np.sqrt(2 * np.pi * 100) * (100 / np.e) ** 100


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


def print_results(desc: str, flops: float) -> None:
    naivest_loop_flops = estimate_flops(naive_loop, int(n), "native")
    flop_time = 1.0 / flops
    print(f"{desc}: {flops:.2E} FLOPS, or {flop_time:.2E} seconds for 1 FLOP")
    print(f"Time for 100x100 Matrix: {(N_FACT/3.15576e+7)*flop_time} years")


if __name__ == "__main__":
    n = 1e6
    naivest_loop_flops = estimate_flops(naive_loop, int(n), "native")
    print_results("Python list, naive for loop", naivest_loop_flops)

    naive_loop_flops = estimate_flops(naive_loop, int(n), "native")
    print_results("NumPy array, naive for loop", naive_loop_flops)

    n = 1e8
    numpy_flops = estimate_flops(numpy_vector_op, int(n), "numpy")
    print_results("NumPy array, vectorized sum", numpy_flops)

    torch_cpu_flops = estimate_flops(torch_cpu_or_gpu, int(n), "torch_cpu")
    print_results("Torch Tensor (CPU), vectorized sum", torch_cpu_flops)

    torch_gpu_flops = estimate_flops(torch_cpu_or_gpu, int(n), "torch_gpu")
    print_results("Torch Tensor (GPU), vectorized sum", torch_gpu_flops)
