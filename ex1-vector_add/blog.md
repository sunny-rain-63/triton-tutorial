摘要：本文在**昇腾 NPU（Atlas A3 / A2）** 上一步步实现向量加算子，练习 `tl.arange`、`tl.load`、`tl.store`、`tl.program_id`、`tl.constexpr`、`tl.num_programs` 这些 Triton 原语，并建立昇腾最重要的编程习惯——**固定核数 + 核内循环**。

> 建议先花 10 分钟看[昇腾 NPU 与 Triton-Ascend 速览](../ascend_basics.md)，知道 UB、AI Core、grid 这几个词是什么意思，这一课会轻松很多。

## 向量加算子实战

入门先学 `a + b`。向量加法可以写成 `向量c = 向量a + 向量b`，也就是把 a 和 b 中**对应位置**的每个数字相加：

```text
a:  [ 1,  2,  3,  4 ]
b:  [10, 20, 30, 40 ]
         ↓ 逐位置相加
c:  [11, 22, 33, 44 ]
```

这个算子简单到没有任何算法难度，正好用来把注意力全放在"Triton 怎么表达并行"上。

### 1、torch 的向量加法

先用 PyTorch 做一遍，当作后面对答案的"标准答案"。用 `torch.randn` 生成随机向量 a、b，直接相加即可。

**昇腾上只有两个地方和 GPU 教程不一样**：第一行要 `import torch_npu`，设备名从 `'cuda'` 换成 `'npu'`。

```Python
import torch
import torch_npu      # 必须 import，它负责把 npu 设备注册进 PyTorch

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    c = a + b
    print(a, b, c, sep="\n")
```

可以得到如下输出，第三个 tensor 的值是前两个 tensor 对应位置相加。由于是随机数据，每次运行结果会变化。

```text
tensor([-0.3947,  0.1963,  0.4782, -0.0215,  1.5055,  0.1066, -0.8224,  0.0999,
        -0.1316,  0.3244, -1.6962, -0.1411,  0.5005,  0.0396,  0.4774,  0.9639],
       device='npu:0')
tensor([-0.1621, -1.0437,  0.5023,  0.3897,  0.6714, -0.8212, -0.2596, -0.3467,
        -2.2264,  0.7489,  1.3961, -2.1076,  0.0119, -0.8835, -0.4079,  1.8599],
       device='npu:0')
tensor([-0.5568, -0.8474,  0.9805,  0.3682,  2.1769, -0.7145, -1.0820, -0.2469,
        -2.3581,  1.0732, -0.3000, -2.2486,  0.5124, -0.8439,  0.0695,  2.8238],
       device='npu:0')
```

注意 `device='npu:0'`，说明数据确实在 NPU 上。这一步跑通，意味着你的 CANN 驱动、`torch_npu` 都是好的。

PyTorch 在昇腾上不是自己算的，它会把 `a + b` 派发给 CANN 的 **aclnn 算子库**（底层是华为写好的高性能 kernel）。我们接下来做的，就是**自己写一个 kernel 来替代它**。

### 2、单 program 16 个元素加法和验证

现在写我们自己的 Triton kernel。

先考虑最简单的情况：**只用 1 个 program 做完全部工作**，也就是 1 个核要完成 16 个元素的计算。

Triton 的 kernel 函数需要加 `@triton.jit` 装饰器，它标记"这是一段要被即时编译（JIT）成 NPU 指令的代码"。然后把 tensor 作为参数传进去——注意**实际传下去的是 tensor 的 `data_ptr()`，也就是一个指针（内存地址）**，而不是 tensor 对象本身。所以 kernel 里的参数我们习惯命名成 `a_ptr`、`b_ptr`。

一个空壳 kernel 长这样：

```Python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    pass

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)                            # 只启动 1 个 program
    vector_add_kernel[grid](a, b, c)
```

`grid = (1,)` 就是"我要启动 1 个 program"。`kernel[grid](...)` 这个方括号语法是 Triton 的启动方式，方括号里放 grid，圆括号里放参数。

kernel 内部要做的事：取出 16 个元素 → 对应位置相加 → 存起来。

- 用 `tl.arange(0, 16)` 生成连续索引 `[0, 1, ..., 15]`；
- a 的地址就可以用 `a_ptr + offsets` 表达——**指针加一个索引向量，得到一个地址向量**，这是 Triton 里最核心的一个表达习惯；
- 用 `tl.load` 把这些地址上的数据搬进来（在昇腾上，这一步就是把数据从 HBM 搬进 **UB 片上内存**）；
- 相加后用 `tl.store` 写回。

```Python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    # 生成连续索引 [0, 1, ..., 15]，用于访问 16 个元素
    offsets = tl.arange(0, 16)
    # 根据索引从 a_ptr 指向的地址加载 16 个元素（HBM → UB）
    a = tl.load(a_ptr + offsets)
    # 根据索引从 b_ptr 指向的地址加载 16 个元素
    b = tl.load(b_ptr + offsets)
    # 对应位置元素相加（在 Vector 核上执行）
    c = a + b
    # 将结果写回到 c_ptr 指向的地址（UB → HBM）
    tl.store(c_ptr + offsets, c)
```

这里有个很重要的认知：`a = tl.load(...)` 里的 `a` **不是一个数，而是一整个长度 16 的向量**。`c = a + b` 一行就完成了 16 个加法。你写的是"整块数据的操作"，编译器负责把它翻译成 Vector 核的 SIMD 指令——这就是 Triton 比手写 C++ 算子省力的地方。

接下来验证。用 `torch.empty_like` 开一块和 a 同形状的空间接结果，然后调用 `solve`：

```Python
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
```

对比答案用 `torch.allclose`（浮点数不能用 `==` 判相等，要允许一点误差）。完整程序如下：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c)

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

运行会得到 `✅ Triton and Torch match`，说明算对了。

> **第一次运行会慢几秒**，因为 Triton 要走完整条编译链（Python → ttir.mlir → ttadapter.mlir → 毕昇编译器 → NPU 二进制）。编译结果会缓存在 `~/.triton/cache`，第二次就快了。

### 3、通过 mask 控制元素访问

如果输入是 15 个元素呢？

在 NVIDIA GPU 上，`tl.arange(0, 15)` 会直接报错 `ValueError: arange's range must be a power of 2`，因为社区 Triton 要求 `arange` 的长度是 2 的幂。

**这是昇腾和 GPU 的一个差异：Triton-Ascend 放宽了这个限制**，`tl.arange(0, 15)` 在昇腾上是合法的。

但是！**我们仍然推荐你按 2 的幂来写**，原因有两个：

1. **对齐**。昇腾的 UB 要求张量尾轴按 32B 对齐（`float32` 即 8 个元素），长度取 2 的幂天然满足，否则硬件会自动补齐，白浪费带宽；
2. **可移植**。同一份 kernel 想在 GPU 上也能跑，就得遵守更严格的那一边。

所以正确的做法不是把 `arange` 改成 15，而是**保持 16，用 mask 把多出来的那一个位置屏蔽掉**。

mask 是 `tl.load` 和 `tl.store` 的一个参数，含义是"这个位置到底要不要真的访问内存"。计算方式就是把索引和真实长度比一下：

```Python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    # 计算 mask：只处理 offsets < 15 的位置
    # mask = [True, True, ..., True, False]，最后一个位置被屏蔽
    mask = offsets < 15
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

没有 mask 会怎样？第 16 个位置会去读**不属于你的那块内存**（越界访问），结果可能是脏数据，也可能直接把程序搞崩。mask 是算子健壮性的第一道防线，**任何时候长度不能被 BLOCK_SIZE 整除，就必须有 mask**。

元素个数不一定是 15，1~16 都有可能，所以把 `N` 作为参数传进去：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    offsets = tl.arange(0, 16)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c, N)

if __name__ == "__main__":
    for N in range(1, 16 + 1):
        a = torch.randn(N, device='npu')
        b = torch.randn(N, device='npu')
        torch_output = a + b
        triton_output = torch.empty_like(a)
        solve(a, b, triton_output, N)
        if torch.allclose(triton_output, torch_output):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
```

运行以上程序会输出 16 个 `✅ Triton and Torch match`，我们的算子通过了第一阶段的健壮性检测。

你可以把 `tl.arange` 里的 16 改大，来支持更大的 N，动手试试。

### 4、多 Block(program) 运行

一直用 1 个 program 有两个问题。

**第一个问题是算不下。** `tl.arange` 的上限是 `1048576`（2^20），再大会报 `ValueError: numel (2097152) exceeds triton maximum tensor numel (1048576)`。

**第二个问题在昇腾上更要紧：UB 装不下。** UB 只有 192KB，而且默认开 double buffer 要留两份副本。按 `float32`（4 字节）算：

```text
单 tile 占用 ≈ BLOCK_SIZE × 4B × 2
BLOCK_SIZE = 65536  →  65536 × 4 × 2 = 512KB  ≫ 192KB   ❌ ub overflow
BLOCK_SIZE = 1024   →  1024 × 4 × 2  = 8KB    ✅ 很宽裕
```

超了的话**编译期**就会失败，报 `ub overflow, requires xxx bits while 1572864 bits available!`（`1572864 bits` 正好是 192KB）。这是初学者在昇腾上最常撞的一个错，看到它就知道"单核一次吃太多了"。

所以必须用**多个 program**，每个只处理一小块。

一个 **program**（也叫 Block）是调度的基本单位，在昇腾上它会被分配到一个物理核上执行。**Grid** 就是 program 的集合，可以是 1D、2D、3D。向量是一维数据，1D grid 就够了——顺便说一句，**昇腾上推荐优先用 1D grid**，因为 2D grid 在昇腾后端会被合并成 1D，`(4, 5)` 和 `(20,)` 是等效的，没必要绕。

kernel 内用 `tl.program_id(axis=0)` 获取当前 program 的编号。可以用 `tl.device_print` 把它打出来看看：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def test_pid_kernel():
    pid = tl.program_id(axis=0)
    tl.device_print('pid', pid)

def solve():
    grid = (2,)
    test_pid_kernel[grid]()

if __name__ == "__main__":
    solve()
```

**昇腾上 `tl.device_print` 需要先开环境变量**，否则不会有输出：

```bash
export TRITON_DEVICE_PRINT=1
python test_pid.py
```

输出会在 `HiIPU Print` 区块里，形如：

```text
-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
=> Vec 0
 pid:
[0]
```

回到我们的算子。每个 program 还是处理 16 个元素，那么需要的 program 数就是 `ceil(N/16)`，用 `triton.cdiv(N, 16)` 计算（cdiv = ceiling division，向上取整除法）。

kernel 里的改动只有两行：拿到自己的编号，算出自己负责的起始位置。

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    # 获取当前 program 在 x 方向上的编号
    pid = tl.program_id(axis=0)
    # 计算当前 program 负责的起始元素索引
    block_start = pid * 16
    # 生成当前 block 内的连续索引 [block_start, ..., block_start+15]
    offsets = block_start + tl.arange(0, 16)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (triton.cdiv(N, 16), )
    vector_add_kernel[grid](a, b, c, N)

if __name__ == "__main__":
    N = 12345
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

每个 program 只看自己那一段，互不干扰——这就是数据并行最朴素的样子。

> **昇腾上要留一个心眼**：`grid` 单个维度不能超过 **65535**（`UINT16_MAX`），超了会报 `coreDim=xxxx can't be greater than UINT16_MAX`。
>
> 按上面 `BLOCK_SIZE=16` 的写法，`N` 超过 `16 × 65535 ≈ 105 万` 就会触发。两种解法：
> 1. **把 `BLOCK_SIZE` 调大**，让 `ceil(N / BLOCK_SIZE) ≤ 65535`，即 `BLOCK_SIZE ≥ ceil(N / 65535)`；
> 2. **用第 6 节的"固定核数 + 核内循环"写法**，grid 和 N 彻底解耦，这也是昇腾上更推荐的做法。

### 5、使用参数化的 BLOCK_SIZE

把 16 这个数字写死在 kernel 里不太好，我们希望它能方便地调。做法是把它作为参数传进来，但**它必须是编译期常量**——因为 `tl.arange(0, BLOCK_SIZE)` 要用它来决定张量的形状，形状在编译时就得确定。声明成常量的方式是加类型标注 `tl.constexpr`：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    N = 12345
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**`BLOCK_SIZE` 在昇腾上怎么选？** 两头都有约束，往中间找：

- **下界**：太小会导致 grid 很大、搬运粒度太碎，核启动开销占比高。而且 `BLOCK_SIZE × 元素字节数` 最好是 32B 的整数倍（`float32` 即 ≥ 8），否则硬件要补齐；
- **上界**：`BLOCK_SIZE × 元素字节数 × 2(double buffer)` 不能超过 192KB，`float32` 大约在 24576 个元素附近，再往上就会 `ub overflow`；
- **实践中** `1024 ~ 8192` 是个不错的起点，取 2 的幂。

不想手调就交给 `triton.autotune`，让它跑一遍自动挑最快的：

```Python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N'],          # N 变化时重新调优
)
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
```

配合环境变量就能看到它最后选了哪个：

```bash
export TRITON_PRINT_AUTOTUNING=1
```

### 6、昇腾的正确姿势：固定核数 + 核内循环

前面 5 节的写法在 GPU 和昇腾上都能跑对，但**在昇腾上不够快**。这一节是本课最有价值的部分。

**问题在哪？** 昇腾的物理核只有几十个（Vector 核数，用接口查）。假设你的卡有 48 个 Vector 核，`N = 1000万`、`BLOCK_SIZE = 1024`，那么：

```text
grid = ceil(10000000 / 1024) = 9766 个 program
物理核 = 48 个
→ 硬件要跑 9766 / 48 ≈ 204 轮
→ 每一轮都有核启动 + 初始化开销
```

GPU 上这样写没问题，因为 SM 多、调度器硬件化、启动开销极低。昇腾上这 204 轮的额外开销会实打实地吃掉性能。

**解法：让 grid 等于物理核数，把"轮次"搬到 kernel 内部用 for 循环完成。** 核启动只发生一次，循环在核内跑，没有反复下发的开销。

先查核数（A3 和 A2 核数不同，不要写死）：

```Python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]   # 纯 Vector 算子用这个
aicore_num = properties["num_aicore"]           # 含 tl.dot 的算子用这个
```

向量加是纯 Vector 算子（只有逐元素加法，没有矩阵乘），所以按 **Vector 核数** 分核：

```Python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 我是第几个核
    pid = tl.program_id(axis=0)
    # 一共启动了几个核（等于 grid 大小）
    num_core = tl.num_programs(axis=0)
    # 整个任务被切成了多少块
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 跨步领任务：pid=0 的核做第 0、num_core、2*num_core… 块
    for block_idx in range(pid, num_blocks, num_core):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    num_core = properties["num_vectorcore"]

    BLOCK_SIZE = 1024
    # grid 固定为物理核数，不再随 N 膨胀
    grid = (min(num_core, triton.cdiv(N, BLOCK_SIZE)), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    N = 12345678
    a = torch.randn(N, device='npu')
    b = torch.randn(N, device='npu')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

几个细节值得注意：

- **`tl.num_programs(axis=0)`** 让 kernel 自己问"一共启动了几个核"，这样 kernel 不用关心外面传了多少，换张卡也不用改代码；
- **为什么跨步而不是连续切段？** 跨步 `range(pid, num_blocks, num_core)` 天然负载均衡。假设 100 块分给 48 个核，跨步方式下每个核拿 2~3 块，差距最多 1 块；要是按"每核连续一大段"去切，除不尽时最后一个核可能明显多干或少干；
- **`min(num_core, cdiv(N, BLOCK_SIZE))`** 是为了 N 很小的时候别启动一堆没活干的空核；
- **grid 和 N 彻底解耦了**，所以第 4 节说的 65535 上限问题自动消失，N 再大也不会超。

> **如果核内一次还是放不下怎么办？** 比如 `BLOCK_SIZE` 本身就超 UB 了，那就在核内**再切一层**：外层循环按 `BLOCK_SIZE` 领任务，内层再按 `BLOCK_SIZE_SUB` 分批搬运。这就是昇腾文档里常提的 `ncore / xblock / xblock_sub` 三级切分思路，到写 softmax、layernorm 这类算子时会经常用到。

### 7、测一下到底快了多少

别用 `time.time()`——它测到的是 Host 侧时间，包含 Python 调用和 kernel 下发开销，对于执行很快的 kernel，这些开销甚至比 kernel 本身还大。昇腾提供了 `do_bench_npu`，拿到的是**纯 Device 侧**执行时间：

```Python
import torch
import torch_npu
import triton
from triton.backends.ascend.testing import do_bench_npu

N = 12345678
a = torch.randn(N, device='npu')
b = torch.randn(N, device='npu')
c = torch.empty_like(a)

# 对比三种实现
ms_torch = do_bench_npu(lambda: a + b)
ms_naive = do_bench_npu(lambda: solve_naive(a, b, c, N))        # 第 5 节写法
ms_core  = do_bench_npu(lambda: solve_core_loop(a, b, c, N))    # 第 6 节写法

print(f"torch          : {ms_torch:.4f} ms")
print(f"grid 跟着 N 涨 : {ms_naive:.4f} ms")
print(f"固定核数+循环  : {ms_core:.4f} ms")
```

如果 `mspti` 不可用，`do_bench_npu` 会自动回退到 `torch_npu.profiler` 路径（功能完整但慢一些），并打印一条告警，这是正常的。

向量加是典型的 **memory-bound（访存受限）** 算子——每个元素只做 1 次加法，却要读 2 次写 1 次内存。所以它的性能天花板由 HBM 带宽决定，优化方向全在"怎么搬得更顺"，而不是"怎么算得更快"。能做到接近 torch（也就是接近 CANN 手写算子库）就已经很不错了。

### 8、完整代码

全部代码已保存在：

- [ex1-vector_add/vector_add_kernel.py](vector_add_kernel.py) —— 第 5 节的基础写法
- [ex1-vector_add/vector_add_core_loop.py](vector_add_core_loop.py) —— 第 6 节的昇腾推荐写法
- [ex1-vector_add/vector_add.py](vector_add.py) —— 验证 + 性能对比入口

直接命令行运行：

```bash
python ex1-vector_add/vector_add.py
```

## 继续练习

试试这几个，都是同一个套路的变体，建议自己先写再看题解：

| 题目 | 练什么 |
|---|---|
| ReLU | `tl.maximum`，最简单的逐元素算子 |
| Leaky ReLU | `tl.where` 条件选择 |
| 矩阵拷贝 | 把二维 shape 拍平成一维来算 |
| 颜色反转（RGBA） | 用取模构造带跳跃的 mask |
| 数组反转 | 一个 program 内同时操作首尾两块 |

后三题的题解见[练习1课后作业题解](../homework/homework1.md)。

写的时候记住昇腾的三条习惯：**grid 固定成核数**、**`BLOCK_SIZE` 控制在 UB 容量内**、**尾块一定要有 mask**。

➡️ 下一课：[矩阵转置算子实践](../ex2-matrix_transpose/blog.md)
