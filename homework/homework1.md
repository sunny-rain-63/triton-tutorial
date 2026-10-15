摘要：练习1的三道课后题题解（昇腾 NPU 版）。这三题都是向量加的变体，重点练 **offsets 的构造**和 **mask 的设计**——这两件事做对了，大部分逐元素算子就都会写了。

> 建议先自己动手写，卡住了再看。每题后面都有"昇腾上的注意点"。

## 练习解答

### 1、Matrix Copy 矩阵拷贝

**题目**：实现一个程序，将输入的 `N×N` 浮点矩阵 `A` 的元素逐一复制到输出矩阵 `B`，即满足 `A[i][j] = B[i][j]`。

**思路**：拷贝不改变元素的位置，所以完全不用管"二维"这件事——把矩阵当成一条长度为 `N*N` 的一维数组处理就行。这是个很有用的简化：**只要读写位置一一对应，二维就可以拍平成一维**。

算出总长度 `n_elements = N * N`，剩下的和向量加一模一样。

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 的起始元素索引
    block_start = pid * BLOCK_SIZE

    # 生成当前 block 内的连续索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask 标记哪些索引有效（不越界）
    mask = offsets < n_elements

    # 根据 offsets 和 mask 从 input_ptr 加载数据
    val = tl.load(input_ptr + offsets, mask=mask)

    # 将结果按照 mask 写回到 output_ptr 对应位置
    tl.store(output_ptr + offsets, val, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    n_elements = N * N
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    copy_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    N = 2
    a = torch.randn((N, N), device='npu')
    torch_output = a.clone()
    triton_output = torch.empty_like(a)
    solve(a, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**昇腾上的注意点**：原始 GPU 题解里常用 `BLOCK_SIZE = 4096`，在昇腾上要算一下 UB：

```text
4096 × 4B(float32) × 2(double buffer) = 32KB   ✅ 还行，没超 192KB
```

这题是安全的。但如果换成更大的 `BLOCK_SIZE`（比如 65536）就会 `ub overflow` 了。这里取 1024 更稳妥，也更接近昇腾上的常用值。

另外 `N` 很大时（`N*N / BLOCK_SIZE > 65535`，即 `N > 8192` 左右）grid 会超上限，这时就该用向量加第 6 节的**固定核数 + 核内循环**写法。

### 2、Color Inversion 颜色反转

**题目**：颜色反转操作——将每个颜色分量 `(R, G, B)` 用 `255 − 分量值` 计算得到，`Alpha` 分量保持不变。

**思路**：图像按 RGBA 存储，**4 个数一组**，最后一个（A）不处理。

一维索引里，A 分量的位置满足 `offsets % 4 == 3`。所以 mask 要同时满足两个条件：

1. 不越界：`offsets < n_pixels * 4`
2. 不是 Alpha 通道：`offsets % 4 != 3`

还有一个容易错的地方：**每个 program 分配的任务数**。一个像素 `pixel` 要处理 4 个数，所以如果用 `BLOCK_SIZE` 表示"像素数"，实际索引范围要乘 4。

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 对图像进行颜色反转（R、G、B 取 255 - 分量值，A 保持不变）
@triton.jit
def invert_kernel(image_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 的起始位置（每个像素 4 个通道，所以要 ×4）
    block_start = pid * BLOCK_SIZE * 4

    # 生成当前 block 内的所有元素索引（BLOCK_SIZE 个像素 × 4 通道）
    offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)

    # mask 条件：
    # 1. offsets < n_pixels*4   → 保证不越界
    # 2. offsets % 4 != 3       → 跳过 Alpha 通道（不做反转）
    mask = (offsets < n_pixels * 4) & (offsets % 4 != 3)

    # 从输入图像中加载数据（只加载有效的 R、G、B 通道）
    input = tl.load(image_ptr + offsets, mask=mask)

    # 颜色反转：output = 255 - input
    output = 255 - input

    # 将反转后的结果写回图像（只写有效的 R、G、B 通道）
    tl.store(image_ptr + offsets, output, mask=mask)


def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 256
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE), )
    invert_kernel[grid](image, n_pixels, BLOCK_SIZE)


if __name__ == "__main__":
    width, height = 1024, 768
    img = torch.randint(0, 256, (width, height, 4), dtype=torch.uint8, device='npu')
    # 拷贝一份
    torch_output = img.clone()
    # 对 R、G、B 通道执行 255 - 值
    torch_output[..., :3] = 255 - torch_output[..., :3]
    triton_output = img.clone()
    solve(triton_output, width, height)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**昇腾上的注意点（这题有两个真坑）**：

1. **`BLOCK_SIZE` 的实际含义被放大了 4 倍**。`tl.arange(0, BLOCK_SIZE * 4)` 才是真实的块大小，算 UB 时别忘乘 4：`BLOCK_SIZE = 256` → 实际 1024 个元素。

2. **`uint8` 在昇腾上不支持**。`tl.load` 在 Ascend 上缺少 `uint8/uint16/uint32/uint64/float64` 的支持（硬件限制）。所以上面这段代码在昇腾上会报类型错误，**需要改成有符号类型**：

   ```Python
   # 用 int16 承载 0~255，并且 255 - x 的中间结果也不会溢出
   img = torch.randint(0, 256, (width, height, 4), dtype=torch.int16, device='npu')
   ```

   这是个很典型的"GPU 代码搬到昇腾要改数据类型"的例子。顺带提一句性能：昇腾的 Vector 比较指令（`CMP`）不支持 `int64/int32`，mask 里如果用了大整数比较，会退化成标量运算拖慢速度——`offsets % 4 != 3` 这种在数据量大时要留意。

### 3、Reverse Array 数组反转

**题目**：反转数组，即 `A[i] ↔ A[N-1-i]`。

**思路**：原地交换。在 Triton 里我们操作的是 `offsets` 这样的连续索引块，所以要在**一个 program 内同时拿到前后两块**：

```text
N = 8
前半：offsets    = [0, 1, 2, 3]
后半：N-1-offsets = [7, 6, 5, 4]
          ↓ 两两交换
```

因为一次交换处理两个位置，所以只需要遍历前半部分，`grid` 数目砍半。如果数组长度为奇数，正中间那个元素本来就不用动（它和自己交换）。

```Python
import torch
import torch_npu
import triton
import triton.language as tl

# 反转数组 即 input[i] ↔ input[N-1-i]
@triton.jit
def reverse_kernel(input_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 内元素的索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 只需要处理数组前半部分，因为每次交换会处理前后两个位置
    mask = offsets < (N // 2)

    # 计算需要交换的前后位置索引
    front_pos = offsets
    back_pos = N - 1 - offsets

    # 从数组中加载这两个位置的值（只加载有效索引）
    front_val = tl.load(input_ptr + front_pos, mask=mask)
    back_val = tl.load(input_ptr + back_pos, mask=mask)

    # 交换前后两个值
    tl.store(input_ptr + front_pos, back_val, mask=mask)   # 前位置写回后值
    tl.store(input_ptr + back_pos, front_val, mask=mask)   # 后位置写回前值


def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE), )
    reverse_kernel[grid](input, N, BLOCK_SIZE)


if __name__ == "__main__":
    N = 21121
    a = torch.randn(N, device='npu')
    torch_output = a.flip(0)
    triton_output = a.clone()
    solve(triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

**昇腾上的注意点**：

1. **`back_pos = N - 1 - offsets` 是递减的**，也就是说读后半部分时，地址是**倒着走**的。昇腾的搬运单元（MTE）对连续正向访问最友好，倒序访问虽然正确但不是最优。想优化的话可以：读后半块时按正序 load 一整块，再用 `tl.flip` 在片上反转，这样两次 load 都是正向连续的。

2. **先 load 完再 store**。上面代码两次 `tl.load` 都在 `tl.store` 之前，这不是偶然——如果先写再读，数据就被覆盖了。`tl.load` 把两块都搬进 UB 之后再写回，天然避免了这个问题。这也是"数据先进片上内存，算完再出去"这个模式的一个好处。

## 小结：三道题的共同套路

```text
1. 算出"总共有多少个元素要处理"            → n_elements
2. 用 pid 和 BLOCK_SIZE 构造 offsets        → 我负责哪些位置
3. 设计 mask                                → 哪些位置真的要访问
   · 越界保护是必需的
   · 业务规则（跳过 Alpha、只处理前半）也放这儿
4. load → 算 → store
```

第 3 步的 mask 设计是这三道题真正的考点。mask 不只是"防越界"的补丁，它是一种**声明式地表达"哪些数据参与运算"** 的手段，比写 `if` 分支更适合 SIMD 硬件。

练完这三道，再回头看向量加第 6 节的**固定核数 + 核内循环**，把这三个 kernel 都改写成那个形式——改完你就真正掌握昇腾上写逐元素算子的标准姿势了。
