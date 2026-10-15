<h3 align="center">
Hands-On Triton Tutorial 📖 · 昇腾 NPU 版
</h3>

<h4 align="center">
在昇腾 Atlas A3 / A2 上学 Triton：不需要 GPU 经验
</h4>

<p align="center">
<a href="README.en.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

本教程面向没有 GPU/NPU 经验的 Triton 初学者，带你从最简单的向量加开始，在**昇腾 NPU（Ascend CANN 架构）** 上一步步写出自己的算子。

这个分支是在 [原教程](https://github.com/OpenMLIR/tt-tut)（NVIDIA GPU 视角）基础上，**针对昇腾 CANN 架构重写**的版本：

- 所有代码改为 `device='npu'` + `import torch_npu`，可直接在 Atlas A3 / A2 上运行；
- 补充了昇腾特有的编程约束：**分核数 = 物理核数**、**UB 192KB 容量限制**、**尾轴 32B 对齐**、`grid ≤ 65535`；
- 每一节都讲清"**为什么**这么写"，而不只是给出能跑的代码；
- 性能数字不照搬别人的结论——给出测量方法（`do_bench_npu`），让你测自己这张卡。

## Triton 简介

[Triton](https://github.com/triton-lang/triton) 是一个让你用 Python 写高性能算子的编程语言（DSL）。它的最大价值是**一份 kernel 代码，多种硬件可跑**：目前有 [NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[华为昇腾](https://github.com/triton-lang/triton-ascend)、[寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)等多个后端。

**优势**：保持 Python 的易用性，同时能把硬件的并行计算和内存层次用起来。比起手写 C++ 算子（昇腾上是 Ascend C），开发效率高一个量级。

**应用**：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

### 昇腾这条路线：Triton-Ascend

在昇腾上，Triton 通过 [**Triton-Ascend**](https://github.com/triton-lang/triton-ascend) 落地。它把你写的 Python kernel 一路降到昇腾 IR，再交给**毕昇编译器**生成 NPU 二进制：

```text
你的 Python kernel
   ↓ triton.jit
kernel.ttir.mlir        ← 平台无关的 Triton IR
   ↓ 昇腾后端适配
kernel.ttadapter.mlir   ← 面向昇腾的适配 IR
   ↓ bishengir-compile（毕昇编译器）
NPU 可执行二进制
```

支持的硬件：**Atlas A2 系列**、**Atlas A3 系列**、**Ascend 950PR & 950DT 系列**。

本教程的代码在 Atlas A3 上验证，A2 同样适用（两者 UB 都是 192KB，差异主要是核数——所以教程里一律用接口查询，不写死）。

## 从哪开始读

```text
第一次接触昇腾？      → 先读「昇腾速览」，10 分钟
写过 CUDA / GPU Triton？→ 直接看「昇腾速览」的第二节和第四节（术语对照表），然后跳到向量加
只想赶紧跑起来？      → 直奔 ex1-vector_add/
```

# 目录

* 一、 [昇腾 NPU 与 Triton-Ascend 速览](ascend_basics.md) ⭐ 建议先读
  * 1、 [昇腾 NPU 长什么样](ascend_basics.md#一昇腾-npu-长什么样够用版)
  * 2、 [`grid` 在昇腾上的含义变了](ascend_basics.md#二grid-在昇腾上的含义变了最重要的一条)
  * 3、 [尾轴对齐与 `ub overflow`](ascend_basics.md#三昇腾特有的两个约束)
  * 4、 [GPU → 昇腾 术语对照表](ascend_basics.md#四gpu--昇腾-术语对照表)
  * 5、 [三个马上能用的调试开关](ascend_basics.md#五三个马上能用的调试开关)
* 二、 [向量加算子实战](ex1-vector_add/blog.md)
  * 1、 [torch 的向量加法](ex1-vector_add/blog.md#1torch-的向量加法)
  * 2、 [单 program 16 个元素加法和验证](ex1-vector_add/blog.md#2单-program-16-个元素加法和验证)
  * 3、 [通过 mask 控制元素访问](ex1-vector_add/blog.md#3通过-mask-控制元素访问)
  * 4、 [多 Block(program) 运行](ex1-vector_add/blog.md#4多-blockprogram-运行)
  * 5、 [使用参数化的 BLOCK_SIZE](ex1-vector_add/blog.md#5使用参数化的-block_size)
  * 6、 [昇腾的正确姿势：固定核数 + 核内循环](ex1-vector_add/blog.md#6昇腾的正确姿势固定核数--核内循环) ⭐
  * 7、 [测一下到底快了多少](ex1-vector_add/blog.md#7测一下到底快了多少)
* 三、 [练习1课后作业题解](homework/homework1.md)
  * 1、 [Matrix Copy 矩阵拷贝](homework/homework1.md#1matrix-copy-矩阵拷贝)
  * 2、 [Color Inversion 颜色反转](homework/homework1.md#2color-inversion-颜色反转)
  * 3、 [Reverse Array 数组反转](homework/homework1.md#3reverse-array-数组反转)
* 四、 [矩阵转置算子实践](ex2-matrix_transpose/blog.md)
  * 1、[1D grid 进行 native 转置](ex2-matrix_transpose/blog.md#11d-grid-进行-native-转置)
  * 2、[2D grid 进行 native 转置](ex2-matrix_transpose/blog.md#22d-grid-进行-native-转置)
  * 3、[tl.trans 原语](ex2-matrix_transpose/blog.md#3tltrans-原语)
  * 4、[维度扩展的 1x1 block 实践](ex2-matrix_transpose/blog.md#4维度扩展的-1x1-block-实践)
  * 5、[通过 mask 控制元素访问](ex2-matrix_transpose/blog.md#5通过-mask-控制元素访问)
  * 6、[昇腾视角：块大小与尾轴对齐](ex2-matrix_transpose/blog.md#6昇腾视角为什么块大小选-32以及尾轴对齐) ⭐
  * 7、[用参数化的 BLOCK_SIZE + 固定核数](ex2-matrix_transpose/blog.md#7用参数化的-block_size--固定核数)

* 附录
  * 附录1、[Triton kernel 原语覆盖情况（含昇腾支持度）](language_cover.md)
  * 附录2、[kernel 优化机制与昇腾编译选项](autotune_cover.md)
  * 附录3、[关联项目及学习资料](other_repo.md)

## 昇腾上写算子的三条铁律

教程里会反复出现，先放这儿：

1. **grid 固定成物理核数**，核内用 `for` 循环吃数据。昇腾只有几十个物理核，GPU 那种"小 block + 海量 grid"的写法会被核启动开销拖垮。
2. **`BLOCK_SIZE` 受 UB 容量约束**。UB 只有 192KB，且默认 double buffer 要留两份。估算：`BLOCK_SIZE × 元素字节数 × 2 ≤ 192KB`，超了编译期直接报 `ub overflow`。
3. **尾轴按 32B 对齐**。`BLOCK_SIZE` 取 2 的幂天然满足（`float32` 需 ≥ 8 个元素，`float16` 需 ≥ 16 个）。不对齐硬件会自动补齐，白费带宽。

## 跑起来

```bash
# 向量加：正确性验证 + 两种分核方式性能对比
python ex1-vector_add/vector_add.py

# 矩阵转置：三种写法对照
python ex2-matrix_transpose/matrix_transpose.py
```

调试用的三个开关（详见[昇腾速览第五节](ascend_basics.md#五三个马上能用的调试开关)）：

```bash
export TRITON_INTERPRET=1      # 在 CPU 上跑 kernel，可以用 breakpoint() 单步
export TRITON_DEVICE_PRINT=1   # 让 tl.device_print / tl.static_print 真的输出
export TRITON_DEBUG=1          # dump 中间 IR 到 ~/.triton/dump
```

## 致谢与来源

原教程作者：[BobHuang](https://github.com/sBobHuang)，[知乎专栏](https://www.zhihu.com/column/c_1948447902964901167)，[Triton 算子开发及编译器资源整理](https://zhuanlan.zhihu.com/p/2018815133590271874)

本昇腾版本的技术依据来自 [Triton-Ascend 官方文档](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh)，特别是：

- [算子开发指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/programming_guide/index.md)（分核、UB、Tiling）
- [GPU 算子迁移指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/migrate_from_gpu.md)
- [架构差异分析](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/architecture_difference.md)
- [调试指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/debugging.md)
- [`do_bench_npu` 性能测试指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/do_bench_npu_guide.md)

> 原仓库已归档停更。本分支为昇腾适配版本，欢迎继续补充 softmax、layernorm、matmul、attention 等进阶算子。
