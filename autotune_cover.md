## kernel 优化机制与昇腾编译选项

本附录整理在昇腾上调一个 Triton 算子时，手上有哪些旋钮。按"**先调分块，再调编译选项**"的顺序看。

### 一、通用优化机制

| 优化机制 | 类型 | 功能说明 | 昇腾上的用法 |
| --- | --- | --- | --- |
| `BLOCK_SIZE` | tile / block 配置 | 每个 program 一次处理的数据量 | **最重要的旋钮**。上限受 UB 192KB 约束：`BLOCK_SIZE × 元素字节 × 2(double buffer) ≤ 192KB`；取 2 的幂以满足 32B 尾轴对齐 |
| `BLOCK_SIZE_SUB` | 核内二级切分 | 核内再切一层，每次只搬一个子块 | UB 吃紧或长序列场景的标准解法，`1024~4096` 起步 |
| `grid`（分核数） | 并行度配置 | 启动多少个 program | 昇腾上**固定成物理核数**：纯 Vector 算子用 `num_vectorcore`，含 `tl.dot` 的用 `num_aicore`。单维 ≤ 65535 |
| `triton.autotune` | 自动调优 | 枚举多组配置实测选最优 | 候选值取 2 的幂；配 `TRITON_PRINT_AUTOTUNING=1` 看选中结果 |
| `num_warps` | 并行度配置 | GPU 上设置每 block 的 warp 数 | **A2/A3 的 SIMD 路径上不起作用**，别照搬 GPU 调参经验（仅 950 的纯 SIMT 模式有效） |

### 二、`triton.autotune` 在昇腾上怎么用

```Python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        # 也可以在 Config 里混入昇腾编译选项
        triton.Config({'BLOCK_SIZE': 4096, 'multibuffer': False}),
    ],
    key=['n_elements'],      # 这个值变化时重新调优
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

配套环境变量：

```bash
export TRITON_PRINT_AUTOTUNING=1   # 打印最终选中的配置
export TRITON_CACHE_AUTOTUNING=1   # 缓存调优结果，避免每次重调
export TRITON_BENCH_METHOD=npu     # 调优时用 do_bench_npu 测纯 Device 时间（短 kernel 强烈建议）
```

> `TRITON_BENCH_METHOD=npu` 很关键：默认的 Host 侧计时在短 kernel 上会被下发开销主导，导致 autotune 选出的"最优"配置其实不是 Device 侧最优。

### 三、进阶：`max_autotune`

昇腾上要榨干性能，除了 `BLOCK_SIZE` 还要调 `num_stages`、`enable_hivm_auto_cv_balance`、`tile_mix_vector_loop` 等一堆硬件相关参数。全部手写进 `configs` 会爆炸，所以 Triton-Ascend 提供了 `max_autotune`：你只给基础配置，它自动生成组合。

```Python
from triton.backends.ascend.runtime import max_autotune

@max_autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
    kernel_type="vector",              # 可选 vector / cube / mix
    enable_ubuf_saving=[True, False],  # 可选，默认已包含
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr, **META):
    ...
```

注意 kernel 签名要加 `**META` 来接收自动注入的编译选项。

### 四、昇腾编译选项速查

这些通过 `kernel[grid](..., 选项=值)` 或 `triton.Config({...})` 传入。

| 选项 | 作用 | 默认值 | 什么时候动它 |
| --- | --- | --- | --- |
| `multibuffer` | 多缓冲总开关，控制"搬运与计算"流水重叠（存算并行） | `True` | **UB overflow 时**可设 `False` 救急，代价是流水变差。优先调小 `BLOCK_SIZE` |
| `num_stages` | 设为 `1` 等于关掉自动多缓冲 | 后端 `2`，`triton.Config` 里 `3` | 同上，和 `multibuffer_mode` 不能同时显式设置 |
| `multibuffer_mode` | 按 `gm`/`l1`/`l0c`/`ub` 分层设置缓冲槽数 | `None` | 精细调优阶段 |
| `compile_mode` | `simd`（SIMD 路径）/ `simd_simt_template` / `simt_only`（仅 950） | `simd_simt_template` | A2/A3 用前两个 |
| `unit_flag` | Cube 搬出的一个优化项 | `None` | 矩阵乘类算子调优 |
| `enable_hivm_auto_cv_balance` | CV 融合场景下自动平衡 Cube 与 Vector 执行 | `None` | CV 融合算子 |
| `tile_mix_vector_loop` | Vector 循环切分份数（2/4/8） | `None` | CV 融合算子 |
| `tile_mix_cube_loop` | Cube 循环切分份数（2/4/8） | `None` | CV 融合算子 |
| `set_workspace_multibuffer` | workspace 多缓冲档位（如 2、4） | `None` | workspace 搬运密集的算子 |
| `enable_graph_optimize` | TTIR 图优化 | `True` | 一般不动 |

> 完整列表见 [环境变量与编译选项参考](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/environment_variable_and_compiler_options_reference.md)。

### 五、调优工作流（推荐顺序）

```text
1. 先跑对                 → torch.allclose 过了再谈性能
2. 分核数固定成物理核数   → num_vectorcore / num_aicore
3. 核内循环吃数据         → range(pid, num_blocks, num_core)
4. autotune 扫 BLOCK_SIZE → 配 TRITON_BENCH_METHOD=npu
5. 还不够快？看 profile   → msprof 采集，看 aiv_mte2_ratio 等指标
6. 针对瓶颈调编译选项     → multibuffer / tile_mix_* / unit_flag
```

### 六、性能排查要看的几个点

出现"算子慢但说不出哪慢"的时候，按这个清单过一遍：

| 症状 | 可能原因 | 怎么查 |
| --- | --- | --- |
| grid 远大于物理核数 | 多轮下发，核启动开销累积 | 打印 grid 和 `num_vectorcore` 对比 |
| 搬运占比异常低 | 存在未映射成 SIMD 的标量操作 | `TRITON_DEBUG=1` dump IR，看 HIVM IR 阶段有没有纯 scalar 搬运/计算 |
| 尾轴不是 32B 倍数 | 硬件自动补齐，白费带宽 | 算 `尾轴元素数 × 元素字节数` 是否能被 32 整除 |
| multi buffer 没使能 | 单次顺序执行、无 Tiling 切分，或 UB 不足 | 加 for 循环切 Tiling；检查 UB 余量 |
| mask 里用了 `int32/int64` 比较 | Vector CMP 不支持这两种，退化成标量 | 必要时 `.to(tl.float32)` 后再比较 |
| `tl.load` 有部分填充 | 默认值填充造成 MTE2 与 Vector 依赖 | 不关心填充值时加 `care_padding=False` |

性能数据采集用 `msprof`，产出 `PROF_*` 目录里的 `op_summary_*.csv`，关注：

- **`aiv_mte2_time`**：从全局内存搬进 UB 的耗时（μs）
- **`aiv_mte2_ratio`**：搬运耗时占算子总耗时的比例

对 memory-bound 算子（向量加、转置），这个比例高是正常的；对 compute-bound 算子（矩阵乘），比例高说明搬运成了瓶颈。

详见 [性能数据采集方法](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/profiling.md)和 [NPU 高性能编程指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/performance_guidelines.md)。
