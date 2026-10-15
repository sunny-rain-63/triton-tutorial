"""矩阵转置算子：三种写法对照（昇腾 NPU 版）。

对应教程 blog.md：
  - solve_naive      : 第 2 节，1 个元素 1 个 program（最慢，且 grid 容易超 65535）
  - solve_block      : 第 5 节，BLOCK_SIZE x BLOCK_SIZE 块 + tl.trans
  - solve_core_loop  : 第 7 节，固定核数 + 核内循环（昇腾推荐）
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.runtime.driver as driver


def get_vectorcore_num() -> int:
    """转置是纯 Vector 算子（无 tl.dot），按 Vector 核数分核。"""
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["num_vectorcore"]


# ---------------------------------------------------------------- 第 2 节写法
@triton.jit
def matrix_transpose_naive_kernel(input_ptr, output_ptr, rows, cols):
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)

    # 旧位置 i * cols + j
    t = tl.load(input_ptr + row_index * cols + col_index)
    # 新位置 j * rows + i
    tl.store(output_ptr + col_index * rows + row_index, t)


def solve_naive(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # 注意：grid = rows * cols，昇腾上 rows*cols > 65535 就会报
    # coreDim can't be greater than UINT16_MAX
    grid = (rows, cols)
    matrix_transpose_naive_kernel[grid](input, output, rows, cols)


# ---------------------------------------------------------------- 第 5 节写法
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols,
                            BLOCK_SIZE: tl.constexpr):
    # 当前块在行 / 列方向的块号
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)

    # 块内行偏移量、列偏移量
    offs_row = row_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = col_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 行偏移 * 列数 + 列偏移 = 一维偏移（[M,1] 与 [1,N] 广播成 [M,N]）
    old_offs = offs_row[:, None] * cols + offs_col[None, :]

    # 边界块用 mask 保护
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)

    # HBM → UB
    block = tl.load(input_ptr + old_offs, mask=mask)

    # 片上整块转置
    transposed_block = tl.trans(block)

    # 新位置：列偏移 * 行数 + 行偏移
    new_offs = offs_col[:, None] * rows + offs_row[None, :]

    # 输出块形状也转置了，mask 同样要转置
    tl.store(output_ptr + new_offs, transposed_block, mask=tl.trans(mask))


def solve_block(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # 32x32 的 float32 块：32*32*4B*2(double buffer) = 8KB，远小于 UB 的 192KB；
    # 尾轴 32*4B = 128B，是 32B 的整数倍，对齐良好。
    BLOCK_SIZE = 32
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](input, output, rows, cols,
                                  BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------- 第 7 节写法
@triton.jit
def matrix_transpose_core_loop_kernel(input_ptr, output_ptr, rows, cols,
                                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_core = tl.num_programs(axis=0)

    # 把二维块网格拍平成一维任务列表
    num_row_blocks = tl.cdiv(rows, BLOCK_SIZE)
    num_col_blocks = tl.cdiv(cols, BLOCK_SIZE)
    num_blocks = num_row_blocks * num_col_blocks

    # 跨步领任务，天然负载均衡
    for block_idx in range(pid, num_blocks, num_core):
        # 一维任务号还原回二维块坐标
        row_index = block_idx // num_col_blocks
        col_index = block_idx % num_col_blocks

        offs_row = row_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_col = col_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        old_offs = offs_row[:, None] * cols + offs_col[None, :]
        mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)

        block = tl.load(input_ptr + old_offs, mask=mask)
        transposed_block = tl.trans(block)

        new_offs = offs_col[:, None] * rows + offs_row[None, :]
        tl.store(output_ptr + new_offs, transposed_block, mask=tl.trans(mask))


def solve_core_loop(input: torch.Tensor, output: torch.Tensor,
                    rows: int, cols: int):
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(rows, BLOCK_SIZE) * triton.cdiv(cols, BLOCK_SIZE)
    # grid 固定为物理核数，不随矩阵大小膨胀
    grid = (min(get_vectorcore_num(), num_blocks), )
    matrix_transpose_core_loop_kernel[grid](input, output, rows, cols,
                                            BLOCK_SIZE=BLOCK_SIZE)


# 默认导出昇腾推荐写法
solve = solve_core_loop
