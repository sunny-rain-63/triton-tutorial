"""矩阵转置验证入口：正确性 + 性能对比。

运行：
    python ex2-matrix_transpose/matrix_transpose.py
"""

import torch
import torch_npu  # noqa: F401

from matrix_transpose_kernel import (
    solve_naive,
    solve_block,
    solve_core_loop,
    get_vectorcore_num,
)


def check(solve_fn, name, rows, cols):
    a = torch.randn((rows, cols), device='npu')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='npu')
    solve_fn(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print(f"✅ [{name}] Triton and Torch match")
    else:
        print(f"❌ [{name}] Triton and Torch differ")


def benchmark(rows, cols):
    try:
        from triton.backends.ascend.testing import do_bench_npu
    except ImportError:
        print("（当前环境没有 do_bench_npu，跳过性能对比）")
        return

    a = torch.randn((rows, cols), device='npu')
    out = torch.empty((cols, rows), device='npu')

    # torch 的 .T 只是改 stride 的视图操作，不搬数据，
    # 所以要 .contiguous() 才是公平的对比
    ms_torch = do_bench_npu(lambda: a.T.contiguous())
    ms_block = do_bench_npu(lambda: solve_block(a, out, rows, cols))
    ms_core = do_bench_npu(lambda: solve_core_loop(a, out, rows, cols))

    print(f"\nshape = ({rows}, {cols})，Vector 核数 = {get_vectorcore_num()}")
    print(f"  torch .T.contiguous()  : {ms_torch:.4f} ms")
    print(f"  32x32 块 + tl.trans    : {ms_block:.4f} ms")
    print(f"  固定核数 + 核内循环    : {ms_core:.4f} ms")
    print("\n转置是纯搬运（memory-bound）算子，性能上限由 HBM 带宽决定。")


if __name__ == "__main__":
    # 小 shape：三种写法都能跑，用来验证正确性
    rows, cols = 63, 72
    check(solve_naive, "1 元素 1 program", rows, cols)
    check(solve_block, "32x32 块 + tl.trans", rows, cols)
    check(solve_core_loop, "固定核数 + 核内循环", rows, cols)

    # 大 shape：注意 solve_naive 在这里会因 grid 超过 65535 而失败，
    # 这正是教程第 1、2 节要说明的问题，所以不再调用它。
    rows, cols = 1921, 3224
    check(solve_block, "32x32 块 + tl.trans（大 shape）", rows, cols)
    check(solve_core_loop, "固定核数 + 核内循环（大 shape）", rows, cols)

    benchmark(rows, cols)
