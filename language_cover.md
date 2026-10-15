## [Triton kernel 原语](https://triton-lang.org/main/python-api/triton.language.html)覆盖情况（昇腾视角）

"首次出现"列表示本教程哪一课用到了它。"昇腾注意点"列记录该原语在 Ascend A2/A3 上和 GPU 的差异——这是本昇腾版新增的内容，查阅时重点看这一列。

> 完整的 API 支持矩阵（含每个 dtype 的支持情况、特殊限制）见 [Triton-Ascend API 文档](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api)。

### 本教程已覆盖

| 原语 | 类型 | 功能 | 示例 | 首次出现 | 昇腾注意点 |
| --- | --- | --- | --- | --- | --- |
| `tl.arange` | 索引 | 生成连续索引向量 | `offsets = tl.arange(0,16)` | 向量加算子 | **长度不必是 2 的幂**（GPU 必须是），但仍建议取 2 的幂以满足 32B 对齐。上限同为 `TRITON_MAX_TENSOR_NUMEL = 1048576`；`start`/`end` 须 ≥ 0，不支持 `int64` |
| `tl.program_id` | 线程/程序 | 获取 kernel program ID | `pid = tl.program_id(0)` | 向量加算子 | grid 单维 ≤ **65535**；2D grid 会被合并成 1D，优先用 1D |
| `tl.num_programs` | 线程/程序 | 获取启动的 program 总数 | `n = tl.num_programs(0)` | 向量加算子（第6节） | 配合"固定核数 + 核内循环"使用，昇腾上的核心惯用法 |
| `tl.load` | 内存 | 从全局内存读到片上内存(UB)，可配 mask | `a = tl.load(a_ptr+offsets, mask=mask)` | 向量加算子 | 不支持 `uint8/16/32/64`、`fp64`（硬件限制）；不支持 `padding_option`；一次搬运量受 UB 192KB 约束 |
| `tl.store` | 内存 | 从 UB 写回全局内存，可配 mask | `tl.store(c_ptr+offsets, c, mask=mask)` | 向量加算子 | 同上的 dtype 限制 |
| `tl.constexpr` | 编译常量 | 声明 kernel 参数为编译期常量 | `BLOCK_SIZE: tl.constexpr` | 向量加算子（第5节） | 决定 tile 形状，因此直接决定 UB 占用 |
| `tl.cdiv` | 数学 | 向上取整除法 | `tl.cdiv(N, BLOCK_SIZE)` | 向量加算子（第6节） | — |
| `tl.trans` | 形状 | 转置维度 | `tl.trans(a)` | 矩阵转置算子 | 支持 `int8/16/32/64`、`uint8`、`fp16/fp32/bf16`、`bool`；**不支持** `uint16/32/64`、`fp64`、`fp8`；不支持维度 > 8 |
| `tl.device_print` | 调试 | 运行时打印张量值 | `tl.device_print("x=", x)` | 向量加算子（第4节） | **必须设 `TRITON_DEVICE_PRINT=1`**，输出在 `HiIPU Print` 区块；张量过长会被截断 |
| `tl.static_print` | 调试 | 编译期打印常量 | `tl.static_print("BS=", BLOCK_SIZE)` | 昇腾速览 | 同样需要 `TRITON_DEVICE_PRINT=1` |

### 常用但本教程尚未覆盖

这些在写 softmax / layernorm / matmul 时会遇到，附上昇腾注意点方便你自学时查：

| 原语 | 类型 | 功能 | 昇腾注意点 |
| --- | --- | --- | --- |
| `tl.dot` | 线性代数 | 矩阵乘 | 用 **Cube 核**，算子变成 CV 类：分核数按 **AI Core 数**（而非 Vector 核数），尾轴对齐要求升级为 **512B**。`TRITON_F32_DEFAULT` 可选 `ieee`/`hf32` |
| `tl.sum` / `tl.max` / `tl.min` | 归约 | 沿轴归约 | 归约是 softmax/layernorm 的核心；注意一次归约的 tile 要装进 UB |
| `tl.where` | 索引 | 条件选择 | mask 类运算，大数据量下注意比较的 dtype（见下） |
| `tl.maximum` / `tl.minimum` / `tl.clamp` | 数学 | 取最值/截断 | **`propagate_nan=tl.PropagateNan.NONE` 会额外生成 NaN 检测逻辑，显著增加 UB 占用**，UB 紧张时避免 |
| `tl.gather` | 索引 | 按索引取值 | 昇腾推荐先把数据整块搬进 UB 再 `tl.gather`，比直接对全局内存做离散 load 快得多 |
| `tl.flip` | 索引 | 沿维度翻转 | 可用来把"倒序访问全局内存"改成"正序 load + 片上翻转" |
| `tl.exp` / `tl.log` / `tl.erf` / `tl.sqrt` | 数学 | 超越函数 | 在 Vector 核上执行 |
| `tl.zeros` / `tl.full` / `tl.zeros_like` | 创建 | 创建张量 | 每个中间张量都占 UB，数量多了会 overflow |
| `tl.reshape` / `tl.permute` / `tl.broadcast_to` / `tl.expand_dims` | 形状 | 形状变换 | 配合"借轴转置"规避尾轴不对齐 |
| `tl.make_block_ptr` | 内存/指针 | 块指针 | 注意 `shape`/`strides` 要如实反映连续性，否则会退化成离散访存（昇腾上性能杀手） |
| `tl.num_programs` | 线程/程序 | — | 已覆盖，见上表 |

### 昇腾独有的扩展原语

社区 Triton 没有、Triton-Ascend 额外提供的，写复杂算子时很有用：

| 接口 | 用途 |
| --- | --- |
| `tl.load(..., care_padding=False)` | mask 只填充部分数据时，跳过默认值填充，**解除 MTE2 与 Vector 的依赖**，提升流水并行度 |
| `extension.insert_slice` / `extract_slice` | 管理 UB 内局部数据，合并多行/取子块（`import triton.language.extra.cann.extension as extension`） |
| `al.sync_block_all` / `sync_block_set` / `sync_block_wait` | 核间同步（需配合 `disable_auto_inject_block_sync=True`） |
| `triton.backends.ascend.testing.do_bench_npu` | 测**纯 Device 侧**执行时间，排除 Host 下发开销 |
| `triton.backends.ascend.runtime.max_autotune` | 昇腾专用调优装饰器，自动组合 `num_stages`、`multibuffer` 等编译选项 |

### 数据类型支持速查（Ascend A2/A3 vs GPU）

| 类型 | 昇腾 | 说明 |
| --- | --- | --- |
| `float32` / `float16` / `bfloat16` | ✅ | 日常主力 |
| `int8` / `int16` / `int32` / `int64` | ✅ | 但 **Vector ADD 不支持 `int64`**、**Vector CMP 不支持 `int64/int32`**，会退化成标量运算拖慢性能。索引/偏移优先用 `int32`，mask 比较必要时转 `float32` |
| `bool` | ✅ | — |
| `uint8` | 部分 | `tl.trans` 支持，但 `tl.load/store` **不支持**——GPU 上的 `uint8` 图像处理代码需改成 `int16` |
| `uint16` / `uint32` / `uint64` | ❌ | 硬件限制 |
| `float64` | ❌ | 硬件限制 |
| `float8e4` / `float8e5` | ❌ | A2/A3 不支持 |
