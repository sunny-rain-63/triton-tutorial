<h3 align="center">
Hands-On Triton Tutorial 📖 · Ascend NPU Edition
</h3>

<h4 align="center">
Learn Triton on Ascend Atlas A3 / A2: No GPU Experience Required
</h4>

<p align="center">
<a><b>English</b></a> | <a href="README.md"><b>中文</b></a>
</p>

This tutorial is for beginners with no GPU/NPU programming experience. It walks you from a simple vector addition all the way to writing your own kernels on **Huawei Ascend NPUs (CANN architecture)**.

This branch is a rewrite of the [original tutorial](https://github.com/OpenMLIR/tt-tut) (which targeted NVIDIA GPUs), adapted for the Ascend CANN architecture:

- All code uses `device='npu'` with `import torch_npu`, runnable on Atlas A3 / A2 as-is.
- Ascend-specific constraints are explained where they matter: **grid size = physical core count**, the **192KB UB (Unified Buffer) budget**, **32B tail-axis alignment**, and the `grid ≤ 65535` limit.
- Every section explains **why** the code looks the way it does, not just what to type.
- No copied-over benchmark numbers — you get the measurement method (`do_bench_npu`) so you can profile your own card.

> The Chinese version ([README.md](README.md)) is the primary and most complete version of this tutorial. The English materials currently cover the vector addition chapter.

## What is Triton

[Triton](https://github.com/triton-lang/triton) is a Python dialect (DSL) that lets you write high-performance kernels in Python instead of C++. Its biggest selling point is that **one kernel runs on many backends**: [NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia), [AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd), [Huawei Ascend](https://github.com/triton-lang/triton-ascend), [Cambricon](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon), [Moore Threads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads), and [MetaX](https://github.com/FlagTree/flagtree/tree/main/third_party/metax).

**Advantages**: Python-like syntax and productivity, while still exposing the hardware's parallelism and memory hierarchy. Compared to hand-writing C++ kernels (Ascend C on this platform), development is an order of magnitude faster.

**Applications**: accelerating deep learning operators and custom operators, improving LLM training and inference performance.

### The Ascend path: Triton-Ascend

On Ascend hardware, Triton is delivered through [**Triton-Ascend**](https://github.com/triton-lang/triton-ascend). It lowers your Python kernel down to Ascend's own IR, then hands it to the **BiSheng compiler** to generate an NPU binary:

```text
Your Python kernel
   ↓ triton.jit
kernel.ttir.mlir        ← platform-independent Triton IR
   ↓ Ascend backend adaptation
kernel.ttadapter.mlir   ← Ascend-specific adapter IR
   ↓ bishengir-compile (BiSheng compiler)
NPU executable binary
```

Supported hardware: **Atlas A2 series**, **Atlas A3 series**, and **Ascend 950PR & 950DT series**.

The code in this tutorial is written for Atlas A3 and works equally well on A2 — both have a 192KB UB, and the main difference is core count, which the tutorial always queries at runtime rather than hard-coding.

## Three rules for writing kernels on Ascend

These come up again and again throughout the tutorial:

1. **Fix the grid to the physical core count** and loop inside the kernel. An Ascend card has only a few dozen physical cores, so the GPU habit of "small blocks + a huge grid" gets crushed by per-launch overhead.
2. **`BLOCK_SIZE` is bounded by the UB.** The UB is only 192KB, and double buffering is on by default (two copies). Rule of thumb: `BLOCK_SIZE × bytes_per_element × 2 ≤ 192KB`. Exceed it and compilation fails with `ub overflow`.
3. **Keep the tail axis 32B-aligned.** Powers of two satisfy this naturally (≥ 8 elements for `float32`, ≥ 16 for `float16`). Misaligned tensors get padded by hardware, wasting bandwidth.

## Contents

* [Ascend NPU & Triton-Ascend primer](ascend_basics.md) (Chinese) — recommended first read
* [Vector Addition](ex1-vector_add-en/blog.md) (English) · [中文](ex1-vector_add/blog.md)
* [Homework 1 solutions](homework/homework1.md) (Chinese)
* [Matrix Transpose](ex2-matrix_transpose/blog.md) (Chinese)
* Appendix 1: [Triton primitive coverage, with Ascend support notes](language_cover.md) (Chinese)
* Appendix 2: [Tuning mechanisms and Ascend compiler options](autotune_cover.md) (Chinese)
* Appendix 3: [Related projects and learning resources](other_repo.md) (Chinese)

## Running the examples

```bash
# Vector addition: correctness check + comparison of two grid strategies
python ex1-vector_add/vector_add.py

# Matrix transpose: three implementations side by side
python ex2-matrix_transpose/matrix_transpose.py
```

Three debugging switches worth knowing:

```bash
export TRITON_INTERPRET=1      # run the kernel on CPU, use breakpoint() to step through
export TRITON_DEVICE_PRINT=1   # required for tl.device_print / tl.static_print to emit output
export TRITON_DEBUG=1          # dump intermediate IR to ~/.triton/dump
```

## Migrating from GPU Triton

| GPU | Ascend |
| --- | --- |
| `device='cuda'` | `device='npu'` (plus `import torch_npu`) |
| `.cuda()` / `.to("cuda")` | `.npu()` / `.to("npu")` |
| `torch.cuda.*` | `torch.npu.*` (drop unnecessary synchronization) |
| SM | AI Core (dozens, not hundreds) |
| CUDA Core | Vector Core (element-wise, reductions) |
| Tensor Core | Cube Core (matrix multiply, `tl.dot`) |
| Shared Memory / L1 | UB, Unified Buffer (192KB) |
| `num_warps` | No effect on the A2/A3 SIMD path |
| Huge logical grid | Fix grid to core count, loop inside the kernel |

See the [GPU migration guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/migrate_from_gpu.md) for the authoritative list.

## Credits

Original tutorial by [BobHuang](https://github.com/sBobHuang) — [OpenMLIR](https://mlir.top).

The Ascend adaptation is based on the [official Triton-Ascend documentation](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh), in particular the programming guide, GPU migration guide, architecture difference analysis, debugging guide, and `do_bench_npu` guide.

> The original repository is archived. This branch is the Ascend-adapted version; contributions covering softmax, layernorm, matmul, and attention are welcome.
