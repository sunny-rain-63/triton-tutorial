"""向量加算子：昇腾推荐写法（固定核数 + 核内循环）。

对应教程 blog.md 第 6 节。

核心思想：昇腾的物理核只有几十个，grid 开到几千上万会带来多轮下发的
核启动开销。把 grid 固定成物理 Vector 核数，让每个核在内部用 for 循环
把属于自己的那些数据块跑完，核启动只发生一次。

副作用（好的）：grid 和 N 彻底解耦，不会再撞 grid <= 65535 的上限。
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.runtime.driver as driver


def get_vectorcore_num() -> int:
    """查询当前卡的 Vector 核数量。

    纯 Vector 算子（逐元素、归约等）按 Vector 核数分核；
    含 tl.dot 的算子按 AI Core 数（properties["num_aicore"]）分核。
    A3 与 A2 核数不同，所以运行时查询，不要写死。
    """
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["num_vectorcore"]


@triton.jit
def vector_add_core_loop_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 我是第几个核
    pid = tl.program_id(axis=0)
    # 一共启动了几个核（等于 grid 大小），由 kernel 自己问，换卡不用改代码
    num_core = tl.num_programs(axis=0)
    # 整个任务被切成多少块
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 跨步领任务：0 号核做第 0、num_core、2*num_core… 块
    # 跨步比"每核连续一大段"更均衡：块数除不尽核数时，每个核最多只多干一块
    for block_idx in range(pid, num_blocks, num_core):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + offsets, c, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_core = get_vectorcore_num()
    # N 很小时别启动一堆没活干的空核
    grid = (min(num_core, triton.cdiv(N, BLOCK_SIZE)), )
    vector_add_core_loop_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
