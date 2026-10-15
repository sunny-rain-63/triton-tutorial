"""向量加算子验证入口：正确性 + 两种分核方式的性能对比。

运行：
    python ex1-vector_add/vector_add.py
"""

import torch
import torch_npu  # noqa: F401

from vector_add_kernel import solve as solve_naive
from vector_add_core_loop import solve as solve_core_loop, get_vectorcore_num


def check(solve_fn, name, N):
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve_fn(a, b, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print(f"✅ [{name}] Triton and Torch match")
    else:
        print(f"❌ [{name}] Triton and Torch differ")


def benchmark(N):
    """用 do_bench_npu 测纯 Device 侧耗时。

    不要用 time.time()，它会把 Host 侧的 kernel 下发开销也算进去。
    """
    try:
        from triton.backends.ascend.testing import do_bench_npu
    except ImportError:
        print("（当前环境没有 do_bench_npu，跳过性能对比）")
        return

    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    c = torch.empty_like(a)

    ms_torch = do_bench_npu(lambda: a + b)
    ms_naive = do_bench_npu(lambda: solve_naive(a, b, c, N))
    ms_core = do_bench_npu(lambda: solve_core_loop(a, b, c, N))

    print(f"\nN = {N}，Vector 核数 = {get_vectorcore_num()}")
    print(f"  torch (CANN 算子库)   : {ms_torch:.4f} ms")
    print(f"  grid 随 N 增长        : {ms_naive:.4f} ms")
    print(f"  固定核数 + 核内循环   : {ms_core:.4f} ms")
    print("\n向量加是 memory-bound 算子，上限由 HBM 带宽决定，"
          "能接近 torch 就已经不错了。")


if __name__ == "__main__":
    N = 12345678
    check(solve_naive, "grid 随 N 增长", N)
    check(solve_core_loop, "固定核数 + 核内循环", N)
    benchmark(N)
