摘要：本文用大白话讲清楚在昇腾（Ascend）NPU 上写 Triton 算子必须先知道的 5 件事：硬件长什么样、`grid` 到底是什么、数据搬到哪里去、为什么会报 `ub overflow`、怎么把 GPU 教程里的词翻译成昇腾的词。读完再去看向量加，会顺很多。

## 零、先回答一个问题：Triton 和昇腾什么关系

Triton 是一门用 Python 写高性能算子的语言（DSL）。你写的 `tl.load`、`tl.store`、`tl.arange` 这些**原语不变**，变的是"编译器把它翻译成哪种硬件的指令"。

昇腾这条路线叫 [**Triton-Ascend**](https://github.com/triton-lang/triton-ascend)：它把 Triton IR 一路降到昇腾自己的 IR，再交给**毕昇编译器**（`bishengir-compile`）生成能在 NPU 上跑的二进制。整条链路大致是：

```text
你的 Python kernel
   ↓ triton.jit（前端）
kernel.ttir.mlir        ← 平台无关的 Triton IR
   ↓ 昇腾后端适配
kernel.ttadapter.mlir   ← 面向昇腾的适配 IR（调试时最常看的一层）
   ↓ bishengir-compile（毕昇编译器）
NPU 可执行二进制
```

所以本教程里 90% 的代码和 GPU 版**一模一样**，差异集中在三处：设备名、分核数、单核一次能吃多少数据。下面逐个讲。

## 一、昇腾 NPU 长什么样（够用版）

你不需要懂芯片设计，只需要记住这张图：

```text
┌──────────────────────── 昇腾 NPU（一张卡）─────────────────────────┐
│                                                                  │
│   HBM / Global Memory（几十 GB，大但慢）  ← torch tensor 住在这里  │
│                              ↑↓  搬运（MTE 指令）                 │
│   ┌──── AI Core #0 ────┐  ┌──── AI Core #1 ────┐     ...          │
│   │  Cube 核  ×1       │  │  Cube 核  ×1       │                  │
│   │   └ 专算矩阵乘      │  │                    │                  │
│   │  Vector 核 ×2      │  │  Vector 核 ×2      │                  │
│   │   └ 算逐元素/归约   │  │                    │                  │
│   │  UB 片上内存 192KB │  │  UB 192KB          │                  │
│   │   └ 小但极快        │  │                    │                  │
│   └────────────────────┘  └────────────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

三个关键点：

1. **核是"物理"的，数量只有几十个**。一个 AI Core 里有 1 个 Cube 核 + 2 个 Vector 核。所以"Vector 核数 = AI Core 数 × 2"。
2. **算数据前必须先把数据搬到 UB**（Unified Buffer，统一缓冲区）。UB 在 Atlas A2 / A3 上都是 **192KB**，Ascend 950 系列是 248KB。它就是你 kernel 里 `tl.load` 出来那些变量真正待的地方。
3. **UB 很小**。192KB 装不下 `float32` 的 6 万个元素（6万×4B=240KB 就超了），再加上编译器默认开了 double buffer（一份数据留两份副本做流水），实际能用的还要再除以 2。这就是后面 `ub overflow` 的根源。

> 查自己机器上到底几个核，跑这段（A3 和 A2 核数不同，别写死）：
>
> ```Python
> import torch
> import torch_npu
> import triton.runtime.driver as driver
>
> device = torch_npu.npu.current_device()
> props = driver.active.utils.get_device_properties(device)
> print("Vector 核数 =", props["num_vectorcore"])   # 纯 Vector 算子按这个分核
> print("AI Core 数 =", props["num_aicore"])        # 带 tl.dot 的算子按这个分核
> ```

## 二、`grid` 在昇腾上的含义变了（最重要的一条）

这是 GPU 老手最容易踩的坑。

| | GPU（NVIDIA） | 昇腾 NPU |
|---|---|---|
| `grid` 是什么 | **逻辑**任务维度，和物理核解耦 | 更贴近**物理核组**的映射 |
| 开 10 万个 block | 完全正常，硬件自己排队调度 | 每多一轮下发都有**核启动开销**，性能被拖垮 |
| 上限 | 基本没有硬限制 | 单维度 ≤ **65535**（`UINT16_MAX`），超了报 `coreDim=xxx can't be greater than UINT16_MAX` |

GPU 的习惯是"**小 block + 海量 grid**"，让硬件去调度。昇腾只有几十个物理核，你下发 10 万个 program，硬件就得跑 10 万 ÷ 核数 轮，每轮都有启动和初始化成本。

所以昇腾上的推荐写法是反过来的——**固定核数 + 核内循环**：

```Python
# ❌ GPU 习惯：grid 跟着数据量线性膨胀
grid = (triton.cdiv(N, BLOCK_SIZE),)      # N 大了就是几十万个 program

# ✅ 昇腾推荐：grid 固定成物理核数，核内用 for 循环吃完所有数据
grid = (NUM_CORE,)                         # 比如 40、48，取决于你的卡
```

kernel 里配套这么写，`range(pid, num_blocks, num_core)` 是个很漂亮的"跨步领任务"写法：

```Python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)              # 我是第几个核
    num_core = tl.num_programs(0)       # 一共启动了几个核
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)   # 活儿总共分成几块

    # 跨步领任务：0 号核拿第 0、num_core、2*num_core… 块
    for block_idx in range(pid, num_blocks, num_core):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
```

为什么用跨步（`pid, pid+num_core, pid+2*num_core…`）而不是连续切一大段？因为跨步天然负载均衡：块数除不尽核数时，余下的块会摊到前几个核上，每个核最多只多干一块。

> **两点补充**：
> - 昇腾后端有 `auto-blockify` 能力，能把"逻辑块很多"的 kernel 自动折叠到物理核上，所以 grid > 65535 现在也能跑。但前提是**各逻辑块之间顺序无关**；而且它只是帮你兜底，自己写成"固定核数 + 核内循环"性能更可控。
> - `grid` 优先用 1D。写成 2D `(4, 5)` 在昇腾上和 1D `(20,)` 是等效的（会被合并成 1D），不像 GPU 上 2D grid 有独立的调度含义。

## 三、昇腾特有的两个约束

### 1. 尾轴要对齐（32B / 512B）

UB 对张量的**最后一个维度**有对齐要求，不够的话硬件会自动补齐，白白浪费搬运带宽和 UB 空间：

| 算子类型 | 用到的计算单元 | 尾轴对齐要求 |
|---|---|---|
| VV 类（纯向量计算，比如 a+b、ReLU） | 只用 Vector 核 | 尾轴字节数能被 **32B** 整除 |
| CV 类（含 `tl.dot` 的矩阵乘、融合算子） | Cube 核 + Vector 核 | 尾轴字节数能被 **512B** 整除 |

换算一下：`float32` 一个 4 字节，32B ÷ 4B = **8 个元素**；`float16` 是 2 字节，对应 **16 个元素**。所以 `BLOCK_SIZE` 取 2 的幂（64、256、1024…）天然对齐，很省心。反过来，`shape=(2048, 3)` 这种尾轴只有 3 个元素的张量在昇腾上会很慢——这也是后面矩阵转置那一课要留意的地方。

### 2. `ub overflow`：单核一次吃太多

单个 tile 搬进 UB 的数据量超了 192KB，**编译期**就会失败（不是运行期）：

```text
triton.compiler.errors.MLIRCompilationError:
error: ub overflow, requires 3072256 bits while 1572864 bits available!
```

`1572864 bits` = 192KB，就是上面说的 UB 容量。按优先级这么治：

1. **把 `BLOCK_SIZE` 按 2 的幂调小**（4096 → 2048 → 1024）；
2. **核内再切一层子块**：加一个 `BLOCK_SIZE_SUB`，用 for 循环每次只处理 `BLOCK_SIZE_SUB` 个元素，任一时刻 UB 里只住着一个子块；
3. 实在紧张再 **关掉 multi buffer**（`multibuffer=False`），代价是搬运和计算的流水重叠变差，一般不建议先动这个。

一个估算公式，写代码前心算一下就能躲掉大部分坑：

```text
单 tile 占用 ≈ tile 元素数 × 每元素字节数 × buffer 份数（默认 2）
```

比如 `BLOCK_SIZE=65536` 的 `float32`：65536 × 4B × 2 = 512KB ≫ 192KB，必超。

## 四、GPU → 昇腾 术语对照表

看别的 Triton 教程（基本都是 NVIDIA 视角）时，按这张表翻译：

| GPU 说法 | 昇腾对应 | 说明 |
|---|---|---|
| `device='cuda'` | `device='npu'` | 还要在 Python 文件里 `import torch_npu` |
| `.cuda()` / `.to("cuda")` | `.npu()` / `.to("npu")` | — |
| `torch.cuda.*` | `torch.npu.*` | stream / event / synchronize 同理，不需要的同步直接删 |
| SM（流多处理器） | AI Core | 数量级从几百降到几十 |
| CUDA Core | Vector 核 | 管逐元素、归约 |
| Tensor Core | Cube 核 | 管矩阵乘（`tl.dot`） |
| Shared Memory / L1 | UB（统一缓冲区，192KB） | `tl.load` 出来的数据住这儿 |
| warp / `num_warps` | 在 A2/A3 的 SIMD 路径上**不起作用** | 别照搬 GPU 的 `num_warps` 调参经验 |
| `grid` 开到几十万 | 固定成核数 + 核内循环 | 见第二节 |

Python 侧迁移清单（照着删改就行）：

```Python
import torch
import torch_npu                     # 【加】昇腾的 PyTorch 适配库，必须 import

# DEVICE = triton.runtime.driver.active.get_active_torch_device()   # 【删】GPU 设备自动发现
a = torch.randn(N, device='npu')     # 【改】cuda → npu
# assert a.device == DEVICE          # 【删】GPU 设备断言
```

> `import torch_npu` 这行只要写了就生效（它会把 `npu` 设备注册进 PyTorch），即使代码里没显式调用它的函数，**也不能删**。

## 五、三个马上能用的调试开关

| 想干什么 | 怎么做 |
|---|---|
| 在 CPU 上跑 kernel，用 `breakpoint()` 单步看中间变量 | `export TRITON_INTERPRET=1`，调完记得 `unset`（CPU 跑会很慢） |
| 打印 kernel 里某个张量的值 | 代码里写 `tl.device_print("x =", x)`，运行前 `export TRITON_DEVICE_PRINT=1` |
| 打印编译期常量（`BLOCK_SIZE` 之类） | `tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)`，同样需要 `TRITON_DEVICE_PRINT=1` |

另外两个排查编译问题时很有用：

```bash
export TRITON_DEBUG=1           # dump 中间 IR 到 ~/.triton/dump
export TRITON_ALWAYS_COMPILE=1  # 跳过缓存，确保每次都真的重新编译
```

测性能别用 `time.time()`（会把 Host 侧的下发开销算进去），用昇腾提供的 `do_bench_npu`，它拿到的是**纯 Device 侧**执行时间：

```Python
from triton.backends.ascend.testing import do_bench_npu

ms = do_bench_npu(lambda: add_kernel[grid](a, b, c, N, BLOCK_SIZE=1024))
print(f"{ms} ms")
```

## 小结

记住这五句话，就可以去写第一个算子了：

1. `cuda` → `npu`，别忘 `import torch_npu`；
2. `grid` 固定成**物理核数**，核内用 `for` 循环吃数据；
3. 数据要先搬进 **192KB 的 UB** 才能算，所以 `BLOCK_SIZE` 不能太大；
4. 尾轴尽量按 **32B** 对齐（`BLOCK_SIZE` 取 2 的幂最省事）；
5. 报 `ub overflow` 就调小 `BLOCK_SIZE` 或核内再切一层。

➡️ 下一课：[向量加算子实战](ex1-vector_add/blog.md)
