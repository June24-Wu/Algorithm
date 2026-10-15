# AI Infra 一面：题目与答案

题目来源：[AIInfraGuide：AI Infra 一面](https://caomaolufei.github.io/AIInfraGuide/interview/ai-infra-%E4%B8%80%E9%9D%A2/)。

原页面共有 7 道题，下面按照原顺序整理。答案、推导、例子和实现是本仓库的学习笔记，并非网站提供的标准答案。中文与英文讲解覆盖相同知识点，两个版本的 Python 实现来自同一份源代码。

## 1. Roofline 模型的基本原理是什么？如何判断算子是否达到计算瓶颈？

Roofline 模型同时考虑算子的计算量和数据搬运量。算子执行速度受到两个上限约束：计算单元的处理能力，以及存储系统提供数据的速度。

```text
F       = floating-point operations; one FMA counts as two operations
Q       = bytes transferred at the memory level being analyzed
I       = F / Q                          [FLOP/byte]
P_peak  = applicable compute ceiling     [FLOP/second]
BW      = sustainable memory bandwidth   [byte/second]

P_roof  = min(P_peak, BW * I)
I_ridge = P_peak / BW
P_real  = F / measured_kernel_time
```

`I` 是算术强度，即每搬运一个字节完成多少计算。在常见的双对数图中，带宽上限 `BW × I` 是斜线，计算上限 `P_peak` 是水平线。交点叫 ridge point。算术强度位于交点左侧时，带宽约束更严格；右侧时，计算约束更严格。这是性能上界，不代表实际实现一定能达到这个速度。[Roofline 原始报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)。

### 用具体数字判断

假设某 GPU 在对应计算精度下的计算上限是 `120 TFLOP/s`，可持续 HBM 带宽是 `1.5 TB/s`。这是示意数字，不是某张 GPU 的硬件规格。

```text
I_ridge = 120 / 1.5 = 80 FLOP/byte

Kernel A:
    I = 20 FLOP/byte
    Roof = min(120, 1.5 * 20) = 30 TFLOP/s
    Measured = 25 TFLOP/s
    HBM bandwidth = 25 / 20 = 1.25 TB/s

Kernel B:
    I = 160 FLOP/byte
    Roof = min(120, 1.5 * 160) = 120 TFLOP/s
    Measured = 110 TFLOP/s
```

A 已经接近带宽上限，进一步优化应优先考虑复用数据、减少搬运字节，而不是增加计算单元。B 接近计算上限，后续收益可能来自减少计算、改变精度或使用更多计算资源。

### 高算术强度不代表计算单元已经跑满

首先必须选择正确的上限：精度、累加类型、Tensor Core 或普通指令、稠密或硬件支持的结构化稀疏，都要与实际算子一致。不能拿普通 FP32 算子的速度去除以低精度稀疏 Tensor Core 峰值。

然后测量 kernel 时间、实际执行的计算和内存流量。分析 HBM roof 时使用 HBM 字节数，分析 L2 roof 时使用 L2 流量，不能混用。结合对应执行流水线计数器，与相同 shape 的优化实现比较。[Nsight Compute Roofline 分析](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)。

如果某算子 `I = 160`，却只有 `15 TFLOP/s`，不能因为它位于计算区就说计算能力已经耗尽。可能是并行度不足、依赖链过长、寄存器溢出、非计算指令过多或同步等待。Occupancy 高或 GPU busy 高，也不等于有效计算接近峰值。

**面试重点：**区分“理论上哪个资源限制它”与“当前实现实际上是否用满了该资源”。

## 2. GPU 卡数成倍扩展后，哪些环节容易成为瓶颈？如何优化？

先明确是强扩展还是弱扩展。**强扩展**保持总工作量不变，卡越多，单卡计算越少；**弱扩展**随卡数增加总工作量，单卡计算可以保持接近。训练中增加 global batch 还可能改变收敛行为，所以 samples/s 提高不一定意味着达到目标精度的时间更短。

可以按关键路径分解迭代时间：

```text
iteration time
    = useful compute
    + communication not hidden by compute
    + exposed input or host stalls
    + synchronization and imbalance
```

这里强调“未被覆盖的开销”。不能把所有通信时间直接加到计算时间上，因为一部分通信与计算重叠。存在同步时，最慢 rank 会拖住整个步骤，因此要比较各 rank 的时间线，而不是只看平均值。

### 按并行方式分析

| 并行方式或环节 | 扩卡后为何变慢 | 检查与优化方向 |
| --- | --- | --- |
| 数据并行训练 | 梯度通信仍然很大，但强扩展使单卡计算减少，通信更难隐藏。 | 测暴露的 all-reduce 时间；使用梯度分桶与 backward 重叠；调 bucket 大小；梯度累积的中间 microbatch 避免不必要的同步。 |
| Tensor Parallel | 每层频繁 collective；decode batch 很小时，计算不足以覆盖通信。 | 检查 collective 延迟与单卡 GEMM shape；优先在高速互联域内组 TP；选择满足容量和延迟需求的最小合理组。 |
| Pipeline Parallel | 阶段不均衡或 microbatch 太少，产生 pipeline bubble。 | 观察各 stage 空闲时间；重新分层、调 microbatch 和流水线调度，同时考虑激活显存与延迟。 |
| Expert Parallel | MoE dispatch 引入 all-to-all；热门 expert 处理更多 token，产生拖尾。 | 看 expert token counts 和网络流量；改善路由均衡、expert 放置或热点副本；容量限制可能改变模型行为。 |
| 参数和状态分片 | 节省显存的同时增加参数 gather、梯度和状态搬运。 | 同时测通信与峰值显存；调整分片组和预取；分得更细不一定更快。 |
| 独立推理副本 | 请求、tokenizer、网关或 scheduler 无法持续供给更多 worker。 | 分开压测前端与模型 worker；看队列、每轮 token 数、cache locality 和 CPU 饱和情况。 |
| 数据与存储 | 更多 worker 争用存储、解码、checkpoint 和元数据服务。 | 对比真实输入与 synthetic input；数据分片、预取和缓存；错峰或分阶段写 checkpoint。 |

DDP 的分桶设计支持 backward 与梯度归约重叠；模型并行与状态分片具有不同的通信模式。[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[Megatron-LM](https://arxiv.org/abs/1909.08053)、[ZeRO](https://arxiv.org/abs/1910.02054)。

### 估算通信下限

对 `p` 个 rank、每个 rank 持有 `S` 字节数据的简化 ring all-reduce：

```text
T_allreduce approximately
    2 * (p - 1) * link_latency
    + 2 * (p - 1) / p * S / effective_link_bandwidth
```

该估计假设理想 ring、流量均衡且没有争用，其他算法可能不同。大消息通常由字节数主导；大量小 collective 可能由延迟主导。GPU 数量翻倍，不代表共享网络上行带宽也翻倍。

用 [NCCL Tests](https://github.com/NVIDIA/nccl-tests) 测量真实消息大小、拓扑和 collective。如果微基准正常而应用通信很慢，要继续排查 rank 到达时间不一致、争用与重叠不足。

```text
Efficiency = throughput_at_p_GPUs
             / (throughput_at_p0_GPUs * p / p0)
```

模型不能单卡运行时，以能够运行的多卡配置 `p0` 为基线。同时报告吞吐与延迟；训练还要关注达到目标质量所需的时间。

**面试重点：**基于计算、显存和通信成本选择并行策略与放置方式，不能只根据卡数预测性能。

## 3. GEMM 一定是计算瓶颈型算子吗？优化的整体思路是什么？

不一定。Shape、精度、数据复用、内存布局和 kernel 实现共同决定瓶颈。

对 `C = A @ B`，其中 `A: (M,K)`、`B: (K,N)`，每个元素占 `s` 字节：

```text
F approximately 2 * M * N * K
Q_min approximately s * (M*K + K*N + M*N)
I_ideal approximately 2*M*N*K / Q_min
```

这是乐观流量估计：在分析的存储层级中，每个输入只读取一次，输出只写一次，且 `beta=0`。旧 C 的读取、tile 重复加载、packing 或独立 epilogue 都会增加流量。[NVIDIA 矩阵乘性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

大规模方阵 `M=N=K=n` 的理想算术强度约为 `2n/(3s)`，随矩阵规模增加而增加。decode 中常见的 `M=1`、`N/K` 很大的 GEMM，则主要读权重，强度接近 `2/s`。两字节权重约为 `1 FLOP/byte`，除非权重已经驻留在更快的缓存里。

很小的 GEMM 还可能受启动开销或并行工作量不足限制，未必用满 HBM。把一个高效 GEMM 分到过多 GPU 上，可能得到多个低效小 GEMM，再加通信。

### 优化顺序

1. **记录真实负载。**包括 M/N/K、batch、dtype、累加类型、stride、转置、后处理融合和 shape 分布。区分 warm/cold；每轮发生的 packing、转换必须计时。
2. **建立库基线。**在语义相同条件下比较 cuBLAS/cuBLASLt，再判断是否值得定制。
3. **用计数器分类瓶颈。**区分计算、带宽、启动开销和并行度不足。
4. **提高数据复用。**使用 shared memory/register tiling；合并访存；减少 bank conflict 和不必要的布局转换。
5. **持续供给计算流水线。**调整 tile、warp、pipeline stage 和寄存器；使用支持的异步加载与计算重叠。大 tile 复用更好，但可能减少驻留 block。
6. **选择合适算术路径。**使用匹配精度和布局的 Tensor Core kernel；关注对齐和 tile 有效覆盖。具体限制依硬件和库版本而定，精度变化必须验证。
7. **处理小规模或不规则场景。**考虑 grouped/batched GEMM；输出 tile 太少时可考虑 split-K，但必须算上归约和 workspace 开销。
8. **评估 epilogue 融合。**融合 bias、activation、residual 可以减少启动和中间张量搬运，但也可能提高寄存器压力、限制 kernel 选择。

[CUDA 最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[CUTLASS 文档](https://docs.nvidia.com/cutlass/latest/overview.html) 解释了这些访存、资源和融合取舍。

带宽受限的 decode 可以用 batching 摊薄权重读取，但要同时评估用户延迟。权重量化只有在节省的搬运超过反量化与 kernel 开销时才会获得实际收益。

**面试重点：**先确定重要 shape 为什么慢，再针对限制资源优化，并验证端到端收益。

## 4. 性能调优中如何定位和检测瓶颈？使用什么工具？

先固定可复现负载和目标指标。Serving 应分别看排队、TTFT、token 间延迟与输出 token 吞吐；定义清楚 SLO，报告符合目标的请求处理速率。训练应在质量条件可比的前提下看 step time 与有效 tokens/s。

### 从系统时间线逐步深入 kernel

1. **固定实验条件。**记录模型版本、软件与硬件、精度、并行组、输入输出长度、并发、cache hit rate。明确 warm/cold，并控制编译与缓存预热。
2. **拆解关键路径。**用应用指标或 NVTX 标注 tokenization、排队、prefill、decode、sampling、transfer、collective。Nsight Systems 用来找空洞、同步和跨阶段依赖。[Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
3. **每次排除一个假设。**例如用已分词 synthetic input 替换真实输入；比较单组与多组 GPU；固定 context，扫描 batch。一次改变一个变量。
4. **查看主要 kernel。**Nsight Compute 用于分析执行流水线、存储层级、stall、spill、occupancy 和 launch shape。Occupancy 是证据，不是独立优化目标。[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)。
5. **提出可验证的改动。**先说明预期哪个计数器和耗时会变化。例如减少重复权重读取，应降低 bytes/token 并改善 decode 时间。
6. **回到不带 profiler 的端到端测试。**检查正确性、延迟分位数、吞吐和不同负载回归。Profiler 的 replay 与开销可能改变并发和耗时。

下面的命令需要替换成实际应用：

```bash
nsys profile --trace=cuda,nvtx --output=timeline python workload.py
```

CUDA launch 是异步的。CPU 计时器只包围 kernel 提交时，通常测到提交时间，而不是执行完成时间。单独测设备操作可使用相应 stream 的 CUDA event，或在定义好的边界同步；端到端使用具有明确完成边界的墙钟计时。不要在真实负载里给每个算子都加同步，否则会破坏重叠。

| 观察 | 需要验证的假设 | 后续检查 |
| --- | --- | --- |
| GPU 空闲且无队列 | 输入负载不足或请求产生器慢 | 增加客户端并发，绕过前端 |
| GPU 空闲但队列很长 | CPU 调度、同步、锁或 KV admission 限制 | 看 CPU/GPU 时间线及 scheduler 决策 |
| HBM 吞吐高，计算不高 | 访存带宽限制 | 估算 bytes/token，检查数据复用 |
| 对应算术流水线接近上限 | 计算饱和 | 检查精度、实际计算量及优化基线 |
| 计算与带宽都低 | 小 shape、依赖、launch 或延迟限制 | 看 stall、kernel 空隙和并行工作量 |
| Collective 占主要时间 | 网络、rank 到达偏斜或重叠不足 | 比较 rank 时间线与 NCCL 基准 |
| 平均延迟正常，尾延迟很高 | 队列干扰、straggler 或显存压力 | 关联请求长度、队列和 cache 状态 |

CPU profiler 查 tokenizer 和主机工作；Nsight Systems 查顺序与重叠；Nsight Compute 查 kernel 内部。看到 collective 等待时，不要立即归因于网络，可能是另一个 rank 更晚到达。

**面试重点：**给出观测证据、可证伪假设、对照实验和最终收益。非关键路径上的 kernel 变快，不一定让应用变快。

## 5. 请介绍 FlashAttention 的原理。

FlashAttention 通过 IO-aware 的执行顺序计算同一个 dense attention。它减少 HBM 与片上存储之间的数据搬运，避免在 HBM 中物化完整的分数和概率矩阵。它是数学意义上的精确 dense attention，但浮点计算顺序不同可能产生小误差。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)。

### 普通 Attention 的流量问题

```text
S = Q @ K.T / sqrt(head_dim)
P = softmax(S)
O = P @ V
```

长度为 L 时，S 和 P 每个 head 都有 L² 个元素。例如 `L=8192`，仅一个两字节分数矩阵就占 `128 MiB`。写入 S、为 softmax 读取 S、再次读取 P，都会产生较大流量；某些实现还使用更高精度的临时值。

### 分块与 online softmax

固定一个 query tile，逐个遍历 K/V tile。每个 query 行维护当前最大值 m、归一化分母 l，以及尚未归一化的加权 V 累加值 u。新分数块为 s：

```text
Initially: m = -infinity, l = 0, u = 0

m_new = max(m, max(s))
alpha = exp(m - m_new)
p     = exp(s - m_new)

l_new = alpha * l + sum(p)
u_new = alpha * u + p @ V_block

After all key/value blocks: output = u / l
```

最大值变化时，必须按 alpha 重新缩放历史累加结果，才能让不同块处于同一个归一化基准。Causal attention 会跳过或屏蔽未来位置。这里假设每行存在可访问 key；全被屏蔽的块或行需要额外处理。Backward 可以借助输入与保存的归一化统计量重算局部值，而不是保存所有概率。[FlashAttention 算法](https://arxiv.org/pdf/2205.14135)。

Dense attention 的主要计算仍然随长度平方增长，FlashAttention 并没有把它变为线性时间。减少 attention 临时存储，也不等于消除了自回归解码的持久 KV cache。

### Prefill 与 decode 要分别衡量

长序列 prefill 往往明显受益，因为平方大小的中间矩阵代价高。单 token decode 没有大方阵分数矩阵，读取历史 KV 和提供足够并行工作可能成为主要问题，因此不能直接套用 prefill 的提速比例。

FlashAttention-2 进一步改善工作划分、并行度并减少部分非 matmul 工作；最佳实现仍依赖硬件和 shape。[FlashAttention-2](https://arxiv.org/abs/2307.08691)。

**面试重点：**讲清避免了哪些数据搬运、online softmax 如何工作，并区分计算复杂度、临时存储与持久 KV 存储。

## 6. C++ 数组越界写入破坏其他结构，保留了 coredump，应该如何排查？

Core 是某一时刻的进程快照，不是执行历史。它能够显示被破坏的状态及最终崩溃位置，却不一定能直接指出早先越界写入的指令。`free()`、容器或虚函数调用处崩溃，可能只是后果。

### 使用精确匹配的现场产物

保留当时的 executable、debug symbols、shared libraries、build ID、输入和配置。不能随便用重新编译的程序分析旧 core，否则栈和对象布局可能误导判断。部分内存映射也可能没有包含在 core 中。[GDB core 文件说明](https://www.sourceware.org/gdb/current/onlinedocs/gdb.html/Files.html)。

Linux 下可以先执行：

```bash
gdb /path/to/exact/executable /path/to/core
```

```gdb
set pagination off
info files
info sharedlibrary
info threads
thread apply all bt full
frame 0
info args
info locals
info registers
x/i $pc
```

识别信号、出错访问与受损对象。查看指针、长度、分配边界和附近内存；必要时用 `p`、`x` 和反汇编。检查所有线程可能的写入和生命周期变化，但要警惕已经损坏的栈产生不可靠回溯。

### 从受害对象寻找可能的写入者

假设固定数组后面的指针字段出现重复模式值，可能提示数组溢出，但必须验证分配布局，不能假定两个对象物理相邻。检查能访问相关分配的代码：

- 循环边界把 `i < size` 写成 `i <= size`。
- `memcpy` 把元素数量当字节，或反过来。
- Tensor stride、shape 乘积、signed/unsigned 转换有误。
- 分配大小计算发生整数溢出。
- 并发 resize、悬空指针或 use-after-free 表现得像越界。

将损坏区间与候选写入范围、输入大小对应起来。某个对象被破坏，本身并不能证明哪个函数有罪。

### 重现后捕获首次非法写入

可以构建带符号和检测的复现版本：

```bash
clang++ -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer repro.cpp -o repro_asan
./repro_asan
```

AddressSanitizer 可以检测许多 heap/stack/global 越界以及 use-after-free，并报告访问位置与相关分配历史。应对受影响代码做 instrumentation，而不是只检测一个无关的 wrapper。它无法追溯旧的、未插桩 core；同一分配内部的字段间覆盖，也可能逃过普通边界检查。[AddressSanitizer 文档](https://clang.llvm.org/docs/AddressSanitizer.html)。

还可以在新的运行中，在受害对象正确初始化后对小字段设置硬件 watchpoint：

```gdb
break initialize_victim
run
# Advance to a point where victim exists and has its expected value.
watch -location victim->field
continue
```

必须重新确定当前运行的地址，旧 core 地址可能因为 ASLR 和分配变化失效。Watchpoint 有尺寸和数量限制，正常写入也可能先触发。冻结的 core 不会继续执行，因此不能靠在 core 上新设 watchpoint 捕获过去的写入。[GDB watchpoint 文档](https://www.sourceware.org/gdb/current/onlinedocs/gdb.html/Set-Watchpoints.html)。

修复后为准确边界及相邻输入大小添加回归检查。无法重现且没有执行记录时，应区分 core 已证明的事实和推测的历史写入者。

**面试重点：**先用 core 还原损坏状态，再用插桩或现场运行捕获原因；不要把崩溃点直接当根因。

## 7. 手写包含 GQA（Grouped Query Attention）的 Attention 模块。

### Query head 如何共享 K/V？

GQA 让多个 query head 共用同一个 key head 和 value head。设 query head 数为 Hq、KV head 数为 Hkv，要求 `Hq % Hkv == 0`：

```text
group_size = Hq / Hkv
kv_head_for_query_head(h) = floor(h / group_size)

Hkv = Hq: Multi-Head Attention
Hkv = 1: Multi-Query Attention
1 < Hkv < Hq: Grouped-Query Attention
```

Q 投影输出 `Hq × Dh` 维，K/V 分别输出 `Hkv × Dh` 维。共享的是 K/V，不是 softmax：每个 query head 仍然独立计算自己的 attention 分布。[GQA 论文](https://arxiv.org/abs/2305.13245)。

例如 8 个 query head、2 个 KV head：

```text
Query heads 0, 1, 2, 3 -> KV head 0
Query heads 4, 5, 6, 7 -> KV head 1
```

持久 KV 显存估计为：

```text
KV bytes = 2 * layers * live_tokens * Hkv * Dh * bytes_per_element
```

同样的 Hq、head size 和 KV 精度下，相比 MHA，KV 存储缩小 `Hq/Hkv` 倍，但总模型显存、总计算量不会按同样比例缩小。减少独立 K/V head 带来表达能力、显存与解码效率的取舍，质量依赖具体训练和配置。

### 完整 NumPy 实现

下面实现 causal self-attention，并支持 KV cache。为突出分组，省略 RoPE、dropout、padding mask 和训练梯度；使用 float64 作为数值参考，生产环境应选择合适的设备精度与融合 kernel。

```text
x:       (B, S, D)
q:       (B, Hkv, group_size, S, Dh)
k/v:     (B, Hkv, P + S, Dh)
scores:  (B, Hkv, group_size, S, P + S)
output:  (B, S, D)
```

K/V 增加一个长度为 1 的 group 维后，广播到本组 query heads。Cache 仍然只保存 Hkv 个头，不会复制成 Hq 份。

{{python-block-1}}

当前调用中局部位置 i 对应全局位置 `past_length + i`，因此只能访问不超过该位置的 key。带历史 cache 且一次输入多个 token 时，普通方形下三角 mask 不足以表达正确范围。

### 可执行的数值检查

将下面代码接在实现后执行。Oracle 显式重复 KV heads，并逐 query 计算，避免与实现使用同一种分组广播逻辑。重复适合小规模验证，在实际 cache 中会浪费显存。

{{python-block-2}}

期望输出：

```text
Passed 9 MHA/GQA/MQA configurations and all cache checks.
```

生产实现还应补上模型要求的位置编码、缓存合适状态的 key、padding/ragged sequence 处理、目标精度验证及优化 grouped-attention kernel。这里显式构造 scores，是便于理解的参考实现，并不是 FlashAttention。

**面试重点：**K/V 使用更少的 head；正确映射 query 到 KV group；每个 query head 独立 softmax；解码时保持 KV cache 紧凑。
