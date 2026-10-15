"""向量加算子：基础写法（grid 随数据量增长）。

对应教程 blog.md 第 5 节。这是从 GPU Triton 教程迁移过来最直接的写法，
在昇腾上能跑对，但当 N 很大时 grid 会膨胀到几千上万个 program，
核启动开销明显。性能更好的写法见 vector_add_core_loop.py。
"""

import torch
import torch_npu  # noqa: F401  必须 import，负责把 npu 设备注册进 PyTorch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的编号（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 program 负责的起始元素索引，每个 program 处理 BLOCK_SIZE 个元素
    block_start = pid * BLOCK_SIZE

    # 生成块内连续索引 [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask 标记哪些索引有效（不越界），尾块必须有它
    mask = offsets < N

    # 从 HBM 搬运数据到片上内存 UB
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # 在 Vector 核上做逐元素相加
    c = a + b

    # 把结果从 UB 写回 HBM，只写 mask 为 True 的位置
    tl.store(c_ptr + offsets, c, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    # BLOCK_SIZE 取 2 的幂，保证尾轴 32B 对齐；
    # 上限受 UB(192KB) 约束：BLOCK_SIZE * 4B * 2(double buffer) <= 192KB
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
