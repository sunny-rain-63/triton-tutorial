摘要：本文在**昇腾 NPU（Atlas A3 / A2）** 上实现矩阵转置算子，练习 `tl.trans` 原语和 2D block 的表达方式，并借这个算子理解昇腾上两个关键的性能问题：**分核粒度**和**尾轴对齐**。

> 前置：[向量加算子实战](../ex1-vector_add/blog.md)、[昇腾 NPU 与 Triton-Ascend 速览](../ascend_basics.md)

## 矩阵转置算子实践

矩阵转置就是把行和列对调：`output[j][i] = input[i][j]`。

```text
input (2×3)            output (3×2)
┌─────────────┐        ┌────────┐
│  1   2   3  │        │  1   4 │
│  4   5   6  │   →    │  2   5 │
└─────────────┘        │  3   6 │
                       └────────┘
```

它没有任何计算量——**一次加法都没有**，纯粹是把数据从一个地方搬到另一个地方。所以这是一个彻底的 **memory-bound（访存受限）** 算子，性能完全取决于"搬得顺不顺"。

这也让它成为理解昇腾访存特性的最佳教材：同样的逻辑，不同的分块方式，性能能差出一两个数量级。

### 一个必须先理解的概念：内存里没有"二维"

`a[i][j]` 看起来是二维的，但内存是一条直线。PyTorch 默认按**行优先（row-major）** 存放，一个 `rows × cols` 的矩阵在内存里是这样排的：

```text
逻辑视图 (2×3)          内存里的实际排列（一维）
┌─────────────┐
│  1   2   3  │   →     [ 1,  2,  3,  4,  5,  6 ]
│  4   5   6  │           ↑              ↑
└─────────────┘        a[0][0]        a[1][0]
                       index 0        index 3
```

所以 `a[i][j]` 的一维索引是 `i * cols + j`。**这个公式是本课所有代码的基础**，后面每一处偏移计算都是它的变形。

转置后的输出矩阵是 `cols × rows`，它的 `out[j][i]` 对应的一维索引是 `j * rows + i`。于是转置的本质就是：

```text
读 input 的 index = i * cols + j
写 output 的 index = j * rows + i
```

### 1、1D grid 进行 native 转置

最朴素的想法：让每个 program 只搬 1 个元素，program 编号直接当成一维索引。

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i] = input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    # 获取当前 program 编号，直接当作输入的一维索引
    pid = tl.program_id(axis=0)

    # 从旧位置取 1 个元素
    t = tl.load(input_ptr + pid)

    # 从一维索引反推行、列坐标
    row_index = pid // cols
    col_index = pid % cols

    # 按转置后的公式算出新位置
    new_index = col_index * rows + row_index

    # 存到新位置
    tl.store(output_ptr + new_index, t)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = (rows * cols, )
    matrix_transpose_kernel[grid](input, output, rows, cols)


if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='npu')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='npu')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

算是算对了，但这个写法在昇腾上**问题很大**，而且是两个独立的问题：

**问题一：grid 直接撞上限。** `grid = rows * cols`，上面 63×72 = 4536 没事，但稍微大一点就完蛋：

```text
rows = cols = 256   →  grid = 65536 > 65535  ❌ coreDim can't be greater than UINT16_MAX
```

一个 256×256 的小矩阵就超了。

**问题二：每个核只搬 1 个元素，彻底浪费硬件。** 昇腾的 Vector 核是 SIMD 的，一条指令能处理一批数据；UB 有 192KB 可以装一大块。你让它一次只搬 1 个 `float32`（4 字节），相当于用卡车运一粒米——核启动和搬运的固定开销远大于有效工作。

这两个问题的根子是同一个：**粒度太细**。后面几节就是在解决它。

### 2、2D grid 进行 native 转置

既然矩阵是二维的，grid 也可以是二维的。把 `grid` 设成 `(rows, cols)`，两个维度分别用 `tl.program_id(0)` 和 `tl.program_id(1)` 取：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i] = input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    # 第 0 维编号，即行坐标
    row_index = tl.program_id(axis=0)
    # 第 1 维编号，即列坐标
    col_index = tl.program_id(axis=1)

    # 旧位置：i * cols + j
    old_index = row_index * cols + col_index
    t = tl.load(input_ptr + old_index)

    # 新位置：j * rows + i
    new_index = col_index * rows + row_index
    tl.store(output_ptr + new_index, t)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = (rows, cols)
    matrix_transpose_kernel[grid](input, output, rows, cols)


if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='npu')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='npu')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

代码可读性确实好了（不用自己做整除取余），但**性能上和第 1 节没有区别**。

> **昇腾上要知道的一件事**：2D grid 在昇腾后端会被**合并成 1D**，`(4, 5)` 和 `(20,)` 是等效的。所以 2D grid 在这里纯粹是"写起来舒服"，并不像 NVIDIA GPU 上那样有独立的硬件调度含义。昇腾官方的建议是**优先用 1D grid**。
>
> 也就是说，第 1 节的 grid 上限问题，第 2 节一个字都没解决：`(256, 256)` 合并后还是 65536，照样超。

真正的解法是让**每个 program 搬一整块**，而不是一个元素。接下来两节就干这件事。

### 3、tl.trans 原语

Triton 提供了 [`tl.trans`](https://triton-lang.org/main/python-api/generated/triton.language.trans.html)，它能直接把一个块转置。官方文档说：

```text
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.
```

翻译过来：`tl.trans` 是对张量做维度重排，不指定 `dims` 时就等于对二维张量做转置。`dims` 可以用元组传，也可以直接传多个参数（`tl.trans(x, 1, 0)` 和 `tl.trans(x, (1, 0))` 等价）。

关键在于它**不是逐元素搬**，而是在片上做整块重排，昇腾后端会把它映射成专门优化过的数据重排指令。这就是我们要的"一次搬一块"。

> **昇腾上的支持情况**：`tl.trans` 支持 `int8/int16/int32/int64/uint8/float16/float32/bfloat16/bool`，不支持 `uint16/uint32/uint64/float64/float8`，且不支持维度高于 8 的转置。日常用的 `float16/float32/bfloat16` 都没问题。

### 4、维度扩展的 1x1 block 实践

要用 `tl.trans`，先得会在 Triton 里表达一个**二维块**。

向量加里我们用 `tl.arange` 做的是 1D block：

```Python
    # 生成连续索引 [0, 1, ..., 15]，用于访问 16 个元素
    offsets = tl.arange(0, 16)
    # 根据索引从 a_ptr 指向的地址加载 16 个元素
    a = tl.load(a_ptr + offsets)
```

二维要靠**维度扩展**（和 PyTorch 的广播是同一套规则）。核心是切片语法里的 `None`：它表示"在这里插入一个长度为 1 的新维度"。

先在 CPU 上跑一下这段，把它彻底搞清楚（不需要 NPU）：

```Python
if __name__ == "__main__":
    import torch
    offs = torch.arange(5)
    print('offs', offs.shape, offs)
    offs_col = offs[:, None]
    print('offs_col', offs_col.shape, offs_col)
    offs_row = offs[None, :]
    print('offs_row', offs_row.shape, offs_row)
```

输出：

```text
offs torch.Size([5]) tensor([0, 1, 2, 3, 4])
offs_col torch.Size([5, 1]) tensor([[0],
        [1],
        [2],
        [3],
        [4]])
offs_row torch.Size([1, 5]) tensor([[0, 1, 2, 3, 4]])
```

`offs[:, None]` 得到 `[5, 1]`（竖着的一列），`offs[None, :]` 得到 `[1, 5]`（横着的一行）。

**为什么这有用？** 因为 `[M, 1]` 和 `[1, N]` 相加时会广播成 `[M, N]`：

```text
offs_row[:, None]  ×  cols  +  offs_col[None, :]
    [M, 1]                        [1, N]
         ↓ 广播
       [M, N] 的偏移矩阵 —— 正好是这个块里每个元素的一维索引
```

这就是 `i * cols + j` 的向量化版本：**行偏移 × 列数 + 列偏移 = 一维偏移**。

```Python
    # 旧 block 内行偏移量
    offs_row = row_index + tl.arange(0, 1)
    # 旧 block 内列偏移量
    offs_col = col_index + tl.arange(0, 1)

    # 类似 row_index * cols + col_index
    # 行偏移量 * 列数 + 列偏移量 = 行偏移 + 列偏移 = 偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]
```

先用 `1x1` 的块把流程跑通（`tl.arange(0, 1)` 就是只有一个元素的"块"），load 出来 → `tl.trans` → store：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i] = input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)
    # 旧 block 内行偏移量
    offs_row = row_index + tl.arange(0, 1)
    # 旧 block 内列偏移量
    offs_col = col_index + tl.arange(0, 1)

    # 行偏移量 * 列数 + 列偏移量 = 偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]

    # 取出旧 Block
    block = tl.load(input_ptr + old_offs)

    # 使用 tl.trans 进行块内转置
    transposed_block = tl.trans(block)

    # 新位置：类似 col_index * rows + row_index
    new_offs = offs_col[:, None] * rows + offs_row[None, :]

    # 存储转置后的 transposed_block
    tl.store(output_ptr + new_offs, transposed_block)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = (rows, cols)
    matrix_transpose_kernel[grid](input, output, rows, cols)


if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='npu')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='npu')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**这一步性能没有任何提升**——块是 1×1 的，`tl.trans` 转置一个单元素等于什么都没干，grid 还是 `rows × cols`。但是流程对了：**块 → 转置 → 写回**。下一节把块放大，性能才真正来。

### 5、通过 mask 控制元素访问

把块从 `1×1` 放大到 `32×32`，一个 program 一次搬 1024 个元素。只需要把 `tl.arange(0, 1)` 改成 `tl.arange(0, 32)`，并且 `program_id` 的含义从"第几个元素"变成"**第几块**"，所以要乘上块大小：

```Python
    # 旧 block 内行偏移量：pid 现在是"块号"，要乘 32
    offs_row = row_index * 32 + tl.arange(0, 32)
    # 旧 block 内列偏移量
    offs_col = col_index * 32 + tl.arange(0, 32)
```

块变大后，边界块就可能超出矩阵范围（比如 63 行按 32 切，第二块只有 31 行是有效的），所以必须加 mask。二维的 mask 要用两个维度的条件相与：

```Python
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
```

**写回时的 mask 要特别小心。** 输出块是转置过的，形状从 `[32, 32]` 的"行×列"变成了"列×行"，所以 mask 也得跟着转置。有两种写法：

```Python
    # 写法一：直接把输入 mask 转置（推荐，不容易写错）
    tl.store(output_ptr + new_offs, transposed_block, mask=tl.trans(mask))

    # 写法二：按输出的视角重新构造一遍，更直观
    mask_out = (offs_col[:, None] < cols) & (offs_row[None, :] < rows)
    tl.store(output_ptr + new_offs, transposed_block, mask=mask_out)
```

两种都对，本教程用写法一。完整代码如下：

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i] = input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    # 当前块在行方向的块号
    row_index = tl.program_id(axis=0)
    # 当前块在列方向的块号
    col_index = tl.program_id(axis=1)

    # 块内行偏移量 / 列偏移量
    offs_row = row_index * 32 + tl.arange(0, 32)
    offs_col = col_index * 32 + tl.arange(0, 32)

    # 行偏移量 * 列数 + 列偏移量 = 一维偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]

    # 边界块需要 mask 保护
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)

    # 取出旧 Block（HBM → UB）
    block = tl.load(input_ptr + old_offs, mask=mask)

    # 片上整块转置
    transposed_block = tl.trans(block)

    # 新位置：列偏移 * 行数 + 行偏移
    new_offs = offs_col[:, None] * rows + offs_row[None, :]

    # 存储转置后的块，mask 同样要转置
    tl.store(output_ptr + new_offs, transposed_block, mask=tl.trans(mask))


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](input, output, rows, cols)


if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='npu')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='npu')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**这一步才是真正的优化。** 相比第 1、2 节：

- grid 从 `rows × cols` 降到 `ceil(rows/32) × ceil(cols/32)`，缩小了 **1024 倍**，65535 上限基本碰不到了；
- 每个 program 一次搬 1024 个元素而不是 1 个，SIMD 和搬运带宽都用上了；
- 转置在片上（UB 内）完成，不是靠对全局内存做零散的跳跃式写入。

### 6、昇腾视角：为什么块大小选 32，以及尾轴对齐

这一节是昇腾特有的内容，GPU 教程里不会讲，但在昇腾上性能影响很大。

**先算一下 32×32 的块占多少 UB：**

```text
32 × 32 × 4B(float32) × 2(double buffer) = 8KB   ≪ 192KB   ✅ 很宽裕
```

既然这么宽裕，为什么不开到 128×128？

```text
128 × 128 × 4B × 2 = 128KB   接近 192KB 上限
256 × 256 × 4B × 2 = 512KB   ❌ ub overflow
```

所以上限在 `128×128` 附近，并且转置过程中可能还有中间张量，留点余量更安全。**`32 ~ 64` 是比较稳妥的起点**，具体用 `autotune` 实测。

**再说尾轴对齐**，这是昇腾上转置这类算子的核心性能点。昇腾的 UB 要求张量的**最后一个维度**（尾轴）按 **32B** 对齐（纯 Vector 算子；含 `tl.dot` 的 CV 类算子要 512B）。换算：

| 数据类型 | 每元素字节 | 32B 对齐对应的元素数 |
|---|---|---|
| `float32` | 4 | 8 |
| `float16` / `bfloat16` | 2 | 16 |
| `int8` | 1 | 32 |

`BLOCK_SIZE = 32` 的 `float32` 块，尾轴是 32 × 4B = 128B，是 32B 的 4 倍，**完美对齐**。这就是取 2 的幂的好处——顺手就对齐了。

**反例值得记住**：如果你要转置的是 `shape = (2048, 3)` 这种尾轴只有 3 个元素（12 字节）的张量，硬件会把每一行从 12B 自动补齐到 32B，相当于 **白搬了 62% 的带宽**。昇腾官方对这种情况有个"借轴转置"的技巧：把长轴裂出一个对齐的因子"借"给短轴，让两个轴都对齐。初学阶段知道有这个坑就行，真遇到了再去查[昇腾编程指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/programming_guide/index.md)。

### 7、用参数化的 BLOCK_SIZE + 固定核数

按向量加的惯例，把 32 这个魔数参数化。同时把向量加第 6 节学到的**固定核数 + 核内循环**用上，让 grid 不再随矩阵大小膨胀：

```Python
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_core = tl.num_programs(axis=0)

    # 把二维块网格拍平成一维任务列表
    num_row_blocks = tl.cdiv(rows, BLOCK_SIZE)
    num_col_blocks = tl.cdiv(cols, BLOCK_SIZE)
    num_blocks = num_row_blocks * num_col_blocks

    # 跨步领任务
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
```

`block_idx // num_col_blocks` 和 `% num_col_blocks` 这一对，就是把一维任务号还原成二维坐标——和第 1 节从 `pid` 反推 `row/col` 是同一个套路，只不过现在的单位是"块"而不是"元素"。**把多维任务拍平成一维、再在核内还原**，是昇腾上处理高维算子的通用手法，后面写 attention、layernorm 都会反复用到。

全部代码已保存在 [ex2-matrix_transpose/matrix_transpose.py](matrix_transpose.py) 和 [ex2-matrix_transpose/matrix_transpose_kernel.py](matrix_transpose_kernel.py)，直接命令行运行：

```bash
python ex2-matrix_transpose/matrix_transpose.py
```

它会依次验证正确性，并用 `do_bench_npu` 对比几种写法和 `torch` 原生实现的耗时。**自己跑出来的数字才是你这张卡的真实数字**，不同代际的卡、不同 shape 差异都很大，所以本文不贴具体毫秒数。你应该能观察到的趋势是：

```text
1 个元素 1 个 program（第 1/2 节）   ≫   32×32 块（第 5 节）   ≳   固定核数 + 核内循环（第 7 节）
        最慢                                  快很多                      还能再好一点
```

想知道时间花在哪了，用 `msprof` 采集，看 `op_summary_*.csv` 里的 `aiv_mte2_time`（从全局内存搬进 UB 的耗时）和 `aiv_mte2_ratio`（搬运占总耗时的比例）。对转置这种纯搬运算子，这个比例应该很高——如果不高，说明时间花在了不该花的地方（比如标量计算没映射成 SIMD 指令）。

## 继续练习

1. **非连续输入**。你拿到的 torch tensor 往往不是 `contiguous` 的（比如它本身就是另一个 tensor 的转置视图）。试试把 `stride` 作为参数传进 kernel，用 `offs_row[:, None] * stride_row + offs_col[None, :] * stride_col` 代替写死的 `* cols`，写出更健壮的算子。

2. **非方块的 block**。`BLOCK_M` 和 `BLOCK_N` 不一定要相等，试试 `64×32` 和 `32×64`，用 `autotune` 跑一遍，看哪个在你的卡上更快。想想为什么会有差异（提示：尾轴对齐和一次搬运的连续长度）。

3. **低精度**。把输入换成 `float16`，重新算一遍 UB 占用和 32B 对齐对应的元素数，看 `BLOCK_SIZE` 的最优值有没有变。

4. **再往下走**。转置练完，建议挑 ReLU / Leaky ReLU 这类更简单的逐元素算子巩固一下，然后直奔 softmax —— 那是第一个需要"核内按行归约 + 多级切分"的算子，昇腾上的技巧会一下子多起来，可以直接参考 [Triton-Ascend 官方样例](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/tutorials)里的 `02-fused-softmax.py`。
