**TL;DR**: This chapter walks through a step-by-step implementation of vector addition on **Ascend NPUs (Atlas A3 / A2)** and introduces the Triton primitives `tl.arange`, `tl.load`, `tl.store`, `tl.program_id`, `tl.constexpr` and `tl.num_programs`. Along the way it establishes the single most important habit for Ascend kernels: **fix the grid to the physical core count and loop inside the kernel.**

> This is the Ascend-adapted version of the original CUDA-oriented chapter. If you have never touched an Ascend NPU, read the [Ascend NPU & Triton-Ascend primer](../ascend_basics.md) (Chinese) first — ten minutes there saves you an hour here.

## Hands-on Triton kernel implementation: Vector Addition

Vector addition is the "hello world" of Triton. It can be written as:

$$
c = a + b
$$

where $a$, $b$ and $c$ are all 1D tensors (vectors) of the same shape:

```text
a:  [ 1,  2,  3,  4 ]
b:  [10, 20, 30, 40 ]
         ↓ element-wise
c:  [11, 22, 33, 44 ]
```

There is zero algorithmic difficulty here, which is exactly the point: all of your attention can go to *how Triton expresses parallelism*, and *how that maps onto Ascend hardware*.

### 1. PyTorch's built-in vector addition

Let's start with PyTorch so we have a reference answer to check against. PyTorch overloads the `+` operator, so addition is just `a + b`.

**Two things differ from the GPU version**: you must `import torch_npu`, and the device string is `'npu'` instead of `'cuda'`.

We’ll start with a tensor of shape `torch.Size([16])`.
(Why not `15` or `17`? There is a reason for choosing `16` — we'll find out shortly.)

```Python
import torch
import torch_npu      # registers the 'npu' device with PyTorch — required
import random
random.seed(42)

N = 16
a = torch.randn(N, device='npu')
b = torch.randn(N, device='npu')
c = a + b
print(a, b, c, sep="\n")
```

Notice that the $i$-th element in $c$ equals the sum of the $i$-th elements of $a$ and $b$:

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

The `device='npu:0'` suffix confirms the data really lives on the NPU. If this snippet runs, your CANN driver and `torch_npu` installation are healthy.

PyTorch does not compute this itself: on Ascend it dispatches `a + b` to CANN's **aclnn operator library**, a set of hand-tuned kernels written by Huawei. What we are about to do is **replace that kernel with our own**.

> On NVIDIA GPUs the equivalent path is the `vectorized_elementwise_kernel` CUDA kernel in [aten/src/ATen/native/cuda/CUDALoops.cuh](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/CUDALoops.cuh#L334). Same idea, different vendor library.

### 2. Triton addition in one program instance

Now the Triton kernel.

Start with the simplest case: **do all the work in a single program**, i.e. one core handles all 16 elements.

A Triton kernel needs the `@triton.jit` decorator, which marks the function as "compile this just-in-time into device instructions". We then pass tensors as arguments — but note that **what actually gets passed down is `tensor.data_ptr()`, a pointer (memory address)**, not the tensor object. That is why the convention is to name the parameters `a_ptr`, `b_ptr`, `c_ptr`.

An empty kernel looks like this:

```Python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    pass

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)                           # launch exactly one program
    vector_add_kernel[grid](a, b, c)
```

The `kernel[grid](...)` square-bracket syntax is how Triton launches a kernel: the grid goes in brackets, the arguments in parentheses.

Inside the kernel we need to: fetch 16 elements, add them pairwise, store the result.

- `tl.arange(0, 16)` generates the contiguous index vector `[0, 1, ..., 15]`;
- the address of `a` is then just `a_ptr + offsets` — **adding an index vector to a pointer yields a vector of addresses**, which is the single most important idiom in Triton;
- `tl.load` brings that data in (on Ascend, this is the move from HBM into the **UB**, the on-chip Unified Buffer);
- `tl.store` writes the result back.

```Python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    # contiguous indices [0, 1, ..., 15] for 16 elements
    offsets = tl.arange(0, 16)
    # load 16 elements from the address a_ptr points to (HBM → UB)
    a = tl.load(a_ptr + offsets)
    # load 16 elements from the address b_ptr points to
    b = tl.load(b_ptr + offsets)
    # element-wise add, executed on a Vector Core
    c = a + b
    # write the result back (UB → HBM)
    tl.store(c_ptr + offsets, c)
```

A crucial mental shift: `a` in `a = tl.load(...)` **is not a scalar, it is an entire 16-element vector**. The single line `c = a + b` performs all 16 additions. You write *whole-block operations*, and the compiler lowers them to SIMD instructions on the Vector Core. That is where Triton's productivity comes from.

Let's validate. Allocate the output with `torch.empty_like` and call `solve`:

```Python
    triton_output = torch.empty_like(a)
    solve(a, b, triton_output, N)
```

Compare with `torch.allclose` (never use `==` on floats — you need a tolerance). The full program:

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

Running this prints `✅ Triton and Torch match`.

> **The first run takes a few seconds** because Triton walks the whole compilation chain: Python → `ttir.mlir` → `ttadapter.mlir` → BiSheng compiler → NPU binary. The result is cached in `~/.triton/cache`, so subsequent runs are fast.

### 3. Bounded element load/store with `mask`

What if the input has 15 elements?

On NVIDIA GPUs, `tl.arange(0, 15)` fails outright:

```text
ValueError: arange's range must be a power of 2
```

Community Triton requires the `arange` length to be a power of two, because many GPU architectures and memory units prefer power-of-two vector lengths, so Triton enforces it for easier alignment.

**Here is an Ascend difference worth remembering: Triton-Ascend relaxes this restriction.** `tl.arange(0, 15)` is legal on Ascend.

But you should **still write powers of two**, for two reasons:

1. **Alignment.** Ascend's UB wants the tail axis of a tensor 32B-aligned (8 elements for `float32`, 16 for `float16`). Powers of two satisfy this for free; otherwise the hardware pads, wasting bandwidth.
2. **Portability.** If you want the same kernel to run on GPUs too, you have to respect the stricter side.

So the right fix is not to change `arange` to 15 — it is to **keep 16 and mask off the extra lane**.

A `mask` is a Triton tensor of booleans that says which lanes should actually touch memory. It is a parameter of both `tl.load` and `tl.store`:

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    offsets = tl.arange(0, 16)

    mask = offsets < N          # size of your input vector

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c, N)

if __name__ == "__main__":
    for N in range(1, 16+1):
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

What happens without a mask? Lane 16 reads memory that **does not belong to you** — out-of-bounds access, producing garbage or a crash. A mask is the first line of defence for kernel robustness: **whenever the length is not divisible by the block size, you need one.**

**TODO**: In the code above, `N` is a runtime variable, yet Triton kernels are JIT-compiled. Explain why `mask = offsets < N` works fine, while `tl.arange(0, N)` raises an error even when `N` is a power of two. Is there a way to write `tl.arange(0, N)` without errors?

The kernel so far only works for inputs ≤ 16. We want it to handle arbitrary sizes — but we don't want a kernel with an absurdly large compile-time tile size, nor a family of kernels with different fixed tile sizes selected at runtime. Both waste hardware resources. What we want is a *templated* kernel that adapts to any input size.

### 4. Multiple program instances for large inputs

**Callback**: the official docs for `tl.arange()` say:

> Returns contiguous values within the half-open interval [start, end). end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 1048576

So `1048576` (2^20) is the ceiling for one block, and larger inputs need multiple blocks.

**On Ascend there is a second, tighter limit that bites much earlier: the UB.** The UB holds only 192KB, and the compiler enables double buffering by default (two copies of a tensor, to overlap data movement with compute). For `float32` (4 bytes):

```text
UB footprint ≈ BLOCK_SIZE × 4B × 2
BLOCK_SIZE = 65536  →  65536 × 4 × 2 = 512KB  ≫ 192KB   ❌ ub overflow
BLOCK_SIZE = 1024   →  1024 × 4 × 2  = 8KB    ✅ comfortable
```

Exceeding it fails at **compile time**, not run time:

```text
error: ub overflow, requires 3072256 bits while 1572864 bits available!
```

`1572864 bits` is exactly 192KB. This is the single most common error newcomers hit on Ascend, and it always means the same thing: *one core is trying to chew too much at once.*

So we need multiple programs, each handling a slice.

A **program** (also called a Block) is the unit of scheduling; on Ascend it is dispatched to a physical core. The **grid** is the collection of programs and can be 1D, 2D or 3D. A vector is one-dimensional, so a 1D grid suffices — and incidentally, **1D is the recommended default on Ascend**, because the backend flattens 2D grids into 1D anyway (`(4, 5)` and `(20,)` are equivalent).

> **TODO**: In CUDA, you think in terms of threads and warps:
>
> ```c
> int tid = threadIdx.x + blockIdx.x * blockDim.x;
> float a = A[tid];
> float b = B[tid];
> C[tid] = a + b;
> ```
>
> Triton instead lets you express the math and data layout at the tile level and leaves threads and warps to the compiler. Explain why warps and threads are hidden abstractions in Triton, and what benefits (and trade-offs) that gives you.
>
> Note that on Ascend the hidden layer is different from NVIDIA's `Grid — Thread Block — Warp — Thread` hierarchy: there are **AI Cores**, each containing one Cube Core (matrix multiply) and two Vector Cores (element-wise and reductions). This is why **`num_warps` has no effect on the A2/A3 SIMD path** — do not carry over GPU `num_warps` tuning intuition.

Inside the kernel, `tl.program_id(axis=0)` returns the current program's index. You can print it with `tl.device_print`:

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

**On Ascend, `tl.device_print` requires an environment variable**, otherwise nothing is emitted:

```bash
export TRITON_DEVICE_PRINT=1
python test_pid.py
```

Output arrives inside a `HiIPU Print` block:

```text
-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
=> Vec 0
 pid:
[0]
```

Back to our kernel. Each program still handles 16 elements, so we need `ceil(N/16)` programs, computed with `triton.cdiv(N, 16)` (cdiv = ceiling division). Only two lines change: get your own index, compute your own starting offset.

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    # index of this program along axis 0
    pid = tl.program_id(axis=0)
    # first element this program is responsible for
    block_start = pid * 16
    # contiguous indices [block_start, ..., block_start+15]
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

This is SPMD (Single Program, Multiple Data): one program text, many instances, each looking only at its own slice.

> **Another Ascend limit to keep in mind**: a single grid dimension cannot exceed **65535** (`UINT16_MAX`), otherwise you get `coreDim=xxxx can't be greater than UINT16_MAX`.
>
> With `BLOCK_SIZE = 16`, that trips as soon as `N > 16 × 65535 ≈ 1.05M`. Two fixes: raise `BLOCK_SIZE` so that `ceil(N / BLOCK_SIZE) ≤ 65535`, or use the fixed-core-count pattern in section 6 — which decouples the grid from `N` entirely.

### 5. A parameterized `BLOCK_SIZE`

Hard-coding `16` is unsatisfying; we want to tune it. Pass it as an argument — but it **must be a compile-time constant**, because `tl.arange(0, BLOCK_SIZE)` determines a tensor shape, and shapes must be known at compile time. The annotation for that is `tl.constexpr`:

```Python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)         # unique ID of this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
```

Note that unlike the GPU version of this tutorial, **we do not pass `num_warps`** — it has no effect on the A2/A3 SIMD path.

**How do you pick `BLOCK_SIZE` on Ascend?** You are squeezed from both sides:

- **Lower bound**: too small means a huge grid, tiny transfers, and launch overhead dominating. Also `BLOCK_SIZE × bytes_per_element` should be a multiple of 32B (≥ 8 elements for `float32`).
- **Upper bound**: `BLOCK_SIZE × bytes_per_element × 2` must stay under 192KB — roughly 24576 `float32` elements, beyond which you get `ub overflow`.
- **In practice**, `1024 – 8192` is a good starting range. Use powers of two.

Rather than tuning by hand, let `triton.autotune` measure for you:

```Python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N'],          # re-tune when N changes
)
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
```

```bash
export TRITON_PRINT_AUTOTUNING=1   # show the winning config
export TRITON_BENCH_METHOD=npu     # tune on device-side time, not host-side
```

### 6. The Ascend way: fix the grid to the core count

Everything so far is correct on both GPUs and Ascend, but **it is not fast on Ascend**. This section is the most valuable part of the chapter.

**What's wrong?** An Ascend card has only a few dozen physical Vector Cores. Suppose yours has 48, with `N = 10,000,000` and `BLOCK_SIZE = 1024`:

```text
grid = ceil(10000000 / 1024) = 9766 programs
physical cores = 48
→ the hardware runs ~204 rounds
→ every round pays core launch + initialization overhead
```

On a GPU this is fine: many SMs, hardware scheduling, negligible launch cost. On Ascend those 204 rounds of overhead eat your performance for real.

**The fix: make the grid equal to the physical core count, and move the "rounds" into a `for` loop inside the kernel.** The launch happens once; the loop runs on-core with no re-dispatch.

First query the core count (A3 and A2 differ — never hard-code it):

```Python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]   # for pure Vector operators
aicore_num = properties["num_aicore"]           # for operators containing tl.dot
```

Vector addition is a pure Vector operator (element-wise only, no matrix multiply), so we size the grid by **Vector Core count**:

```Python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)              # which core am I
    num_core = tl.num_programs(axis=0)       # how many cores were launched
    num_blocks = tl.cdiv(N, BLOCK_SIZE)      # how many blocks of work exist

    # strided task assignment: core 0 takes blocks 0, num_core, 2*num_core, ...
    for block_idx in range(pid, num_blocks, num_core):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    # grid is pinned to the hardware, it no longer grows with N
    grid = (min(num_core, triton.cdiv(N, BLOCK_SIZE)), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
```

Details worth noting:

- **`tl.num_programs(axis=0)`** lets the kernel ask how many cores were launched, so the same kernel works on a different card without edits.
- **Why strided instead of one contiguous chunk per core?** `range(pid, num_blocks, num_core)` is naturally load-balanced: with 100 blocks over 48 cores, each core gets 2–3 blocks and the imbalance is at most one block. Giving each core one big contiguous range makes the leftovers lumpy.
- **`min(num_core, cdiv(N, BLOCK_SIZE))`** avoids launching idle cores for small `N`.
- **The grid is now decoupled from `N`**, so the 65535 limit from section 4 disappears for good.

> **What if a single block still doesn't fit in UB?** Add a second level: the outer loop claims work in `BLOCK_SIZE` units, an inner loop moves `BLOCK_SIZE_SUB` elements at a time. This is the `ncore / xblock / xblock_sub` three-level tiling pattern you will see throughout the Ascend docs, and it becomes essential for softmax and layernorm.

### 7. Measuring performance correctly

Do **not** use `time.time()` or hand-rolled event timing — those include host-side dispatch overhead, which for short kernels can exceed the kernel itself. Ascend ships `do_bench_npu`, which reports **pure device-side** time:

```Python
import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

N = 12345678
a = torch.randn(N, device='npu')
b = torch.randn(N, device='npu')
c = torch.empty_like(a)

ms_torch = do_bench_npu(lambda: a + b)
ms_naive = do_bench_npu(lambda: solve_naive(a, b, c, N))       # section 5
ms_core  = do_bench_npu(lambda: solve_core_loop(a, b, c, N))   # section 6

print(f"torch (CANN library)  : {ms_torch:.4f} ms")
print(f"grid grows with N     : {ms_naive:.4f} ms")
print(f"fixed cores + loop    : {ms_core:.4f} ms")
```

Useful options: `clear_l2_cache=True` for worst-case/cold-cache numbers, and `target_kernel_name="..."` to isolate one kernel when `fn` launches several. If `mspti` is unavailable, `do_bench_npu` prints a warning and falls back to the `torch_npu.profiler` path — slower but fully functional.

**Numbers are deliberately not quoted in this tutorial.** They depend on your card generation, shape and dtype, so measure your own. The trend you should observe is:

```text
tiny blocks, grid ∝ N  ≫  BLOCK_SIZE=1024, grid ∝ N  ≳  fixed cores + in-kernel loop
     slowest                      much better                   best
```

Vector addition is a textbook **memory-bound** operator: one addition per element, but two reads and one write. Its ceiling is set by HBM bandwidth, so all optimization is about moving data smoothly, not computing faster. Getting close to `torch` (i.e. close to Huawei's hand-tuned library) is already a good result.

**TODO**:

1. **Fixed `N`, varying `BLOCK_SIZE`**: how does `BLOCK_SIZE` affect efficiency, and what is the optimum on your card? At what value do you hit `ub overflow`? Does the predicted value (`192KB / (4B × 2)`) match what you observe?
2. **Fixed `BLOCK_SIZE`, varying `N`**: how does input size affect efficiency? At which `N` does the section 5 kernel exceed the 65535 grid limit?
3. Do your conclusions change between the section 5 and section 6 kernels? Which one is more robust across shapes, and why?

> The original CUDA edition of this chapter pointed to [LeetGPU](https://leetgpu.com) for online judging. That platform offers NVIDIA GPUs only, so for Ascend you verify locally with `torch.allclose` against the PyTorch reference (as all examples here do). The problem statements are still useful as exercises.

### 8. Source Code

- [`ex1-vector_add/vector_add_kernel.py`](../ex1-vector_add/vector_add_kernel.py) — the section 5 baseline
- [`ex1-vector_add/vector_add_core_loop.py`](../ex1-vector_add/vector_add_core_loop.py) — the section 6 Ascend-recommended version
- [`ex1-vector_add/vector_add.py`](../ex1-vector_add/vector_add.py) — correctness check and benchmark entry point

```bash
python ex1-vector_add/vector_add.py
```

A notebook version of this chapter is in [`vector_add.ipynb`](vector_add.ipynb).

## Beyond Vector Addition

More practice, all variations on the same pattern:

| Exercise | What it teaches |
| --- | --- |
| ReLU | `tl.maximum`, the simplest element-wise operator |
| Leaky ReLU | `tl.where` for conditional selection |
| Matrix Copy | flattening a 2D shape into 1D |
| Color Inversion (RGBA) | building a strided mask with modulo — note `uint8` is **not** supported on Ascend, use `int16` |
| Reverse Array | handling two blocks (head and tail) in one program |

Solutions for the last three are in [homework1.md](../homework/homework1.md) (Chinese).

Then continue to [Matrix Transpose](../ex2-matrix_transpose/blog.md) (Chinese), where the tail-axis alignment rule starts to matter a lot.
