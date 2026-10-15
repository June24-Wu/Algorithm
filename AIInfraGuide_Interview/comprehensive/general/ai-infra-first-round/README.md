# AI Infra First Round: Questions and Answers

Question source: [AIInfraGuide: AI Infra First Round](https://caomaolufei.github.io/AIInfraGuide/interview/ai-infra-%E4%B8%80%E9%9D%A2/).

The source page contains seven questions about operator optimization and debugging. The questions below are translated and paraphrased in their original order. The answers, worked examples, and implementation are independent study material, not answers supplied by that website.

## 1. How does the Roofline model work, and how can it identify a compute bottleneck?

The Roofline model relates the work a kernel performs to the data it moves. A kernel cannot execute faster than either its compute resources or its memory system can support:

```text
F       = floating-point operations, counting one FMA as two operations
Q       = bytes transferred at the memory level being analyzed
I       = F / Q                          [FLOP/byte]
P_peak  = applicable compute ceiling     [FLOP/second]
BW      = sustainable memory bandwidth   [byte/second]

P_roof  = min(P_peak, BW * I)
I_ridge = P_peak / BW
P_real  = F / measured_kernel_time
```

On the usual log-log plot, `BW * I` is the sloped bandwidth roof and `P_peak` is the horizontal compute roof. Below the ridge intensity, memory traffic sets the tighter bound; above it, computation sets the tighter bound. This is a bound, not a prediction that an implementation will reach it. [Original Roofline report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf).

### A numerical example

Consider an illustrative GPU with a compute ceiling of `120 TFLOP/s` and sustainable HBM bandwidth of `1.5 TB/s`. These are assumed values, not specifications for a particular GPU.

```text
Ridge intensity = 120 / 1.5 = 80 FLOP/byte

Kernel A:
    I = 20 FLOP/byte
    Roof = min(120, 1.5 * 20) = 30 TFLOP/s
    Measured = 25 TFLOP/s
    Measured HBM bandwidth = 25 / 20 = 1.25 TB/s

Kernel B:
    I = 160 FLOP/byte
    Roof = min(120, 1.5 * 160) = 120 TFLOP/s
    Measured = 110 TFLOP/s
```

Kernel A approaches the bandwidth roof. Reusing data or reducing bytes is more promising than adding arithmetic units. Kernel B approaches the compute roof, so further gains likely require less work, a different precision, or more compute capacity.

### How to establish that computation is actually saturated

First match the roof to the kernel: precision, accumulation type, Tensor Core versus scalar instructions, and dense versus supported structured-sparse computation. An FP32 scalar kernel should not be compared against an unrelated low-precision sparse Tensor Core peak.

Then measure kernel time, executed operations, and memory traffic at the same memory level. An HBM roof uses HBM traffic; an L2 roof uses L2 traffic. Check the relevant execution-pipeline counters and a comparable optimized implementation. Nsight Compute provides Roofline and memory-analysis views for this purpose. [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts).

High arithmetic intensity alone does **not** establish compute saturation. If a kernel with `I = 160` achieves only `15 TFLOP/s` in this example, investigate insufficient parallelism, dependency chains, register spills, instruction overhead, or synchronization. Neither a high occupancy percentage nor a busy-GPU percentage proves useful compute is near its ceiling.

**Interview takeaway:** separate the resource that theoretically limits a workload from the resource the current implementation actually saturates.

## 2. What can become the bottleneck when the GPU count doubles, and how would you address it?

First clarify the scaling experiment. With **strong scaling**, total work stays fixed, so each GPU receives less computation. With **weak scaling**, work grows with GPU count, so per-GPU compute can remain similar. In training, increasing global batch also changes optimization behavior; faster samples per second need not mean faster convergence to the target quality.

A useful critical-path decomposition is:

```text
iteration time
    = useful compute
    + communication not hidden by compute
    + exposed input or host stalls
    + synchronization and imbalance
```

These are exposed contributions: adding every rank's full communication time would double-count overlap. A synchronization-heavy step is constrained by its slowest participating rank, so compare rank timelines rather than only averages.

### The limiting resource depends on the parallelism strategy

| Strategy or subsystem | Why scaling can expose a bottleneck | What to measure and change |
| --- | --- | --- |
| Data parallel training | Gradient communication remains substantial while strong scaling reduces local compute. | Measure exposed all-reduce time. Overlap gradient buckets with backward computation; tune bucket sizes. With accumulation, avoid redundant synchronization on intermediate microbatches. |
| Tensor parallelism | Layers introduce frequent collectives; tiny decode batches leave little compute to hide them. | Inspect collective latency and per-rank GEMM shapes. Keep tightly coupled ranks on fast links and use the smallest parallel group that meets memory and latency requirements. |
| Pipeline parallelism | Uneven stages and too few microbatches create bubbles. | Measure stage idle time. Rebalance stages, tune microbatching, and compare pipeline schedules. Account for activation memory and latency. |
| Expert parallelism | MoE dispatch creates all-to-all traffic, and popular experts become stragglers. | Inspect expert token counts and network traffic. Improve routing balance, placement, or replication of hot experts. Capacity limits can change model behavior. |
| Parameter/state sharding | Memory savings introduce parameter gathers and gradient/state transfers. | Measure actual communication and peak memory. Tune sharding groups and prefetching; do not assume more sharding always improves speed. |
| Independent inference replicas | More replicas do not help if requests, tokenization, networking, or the scheduler cannot feed them. | Load-test the frontend and worker separately. Check queue length, tokens scheduled per iteration, cache locality, and CPU saturation. |
| Shared data and storage | More workers contend for storage, decoding, checkpointing, or metadata services. | Compare real input with synthetic input. Shard data, prefetch, cache, and stagger or stage checkpoint writes. |

DDP exposes bucket controls for overlapping reductions with backward computation; model parallelism and state sharding have different communication patterns. [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [Megatron-LM paper](https://arxiv.org/abs/1909.08053), [ZeRO paper](https://arxiv.org/abs/1910.02054).

### Estimate the communication floor

For a simplified ring all-reduce with `p` ranks and an `S`-byte payload per rank:

```text
T_allreduce approximately
    2 * (p - 1) * link_latency
    + 2 * (p - 1) / p * S / effective_link_bandwidth
```

This idealized estimate assumes ring stages, balanced traffic, and no contention. Other collective algorithms differ. For large messages, bytes dominate; for many small collectives, latency dominates. Doubling GPU count does not double the bandwidth of a shared network uplink.

Benchmark the actual collective, message sizes, and placement with [NCCL Tests](https://github.com/NVIDIA/nccl-tests). A fast microbenchmark with slow application collectives suggests arrival skew, contention, or missed overlap rather than necessarily a bad link.

Measure scaling relative to a feasible baseline:

```text
Efficiency = throughput_at_p_GPUs
             / (throughput_at_p0_GPUs * p / p0)
```

If the model cannot fit on one GPU, use a working multi-GPU group as `p0`. Report both throughput and latency, and for training also time to the target quality.

**Interview takeaway:** choose the parallel strategy and placement from measured compute, memory, and communication costs. GPU count by itself is not a performance model.

## 3. Is GEMM always compute-bound, and how would you optimize it?

No. The shape, precision, reuse, memory layout, and kernel implementation determine the bottleneck.

For `C = A @ B`, with `A: (M, K)`, `B: (K, N)`, and element size `s` bytes:

```text
F approximately 2 * M * N * K
Q_min approximately s * (M*K + K*N + M*N)
I_ideal approximately 2*M*N*K / Q_min
```

This optimistic traffic estimate assumes each input is fetched once at the analyzed memory level and output is written once, with `beta = 0`. Reading old `C`, rereading tiles, packing, or separate epilogues increases traffic. [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

For a large square GEMM with `M = N = K = n`, the estimate becomes `2n / (3s)`: reuse grows with matrix size. For a decode-like GEMM with `M = 1` and large `N, K`, weight reads dominate and intensity approaches `2/s`. With two-byte weights, that is roughly one FLOP per byte unless weights are already resident in a faster cache.

Small GEMMs can also be limited by launch overhead or insufficient parallel work rather than saturated HBM. Splitting one good GEMM across too many GPUs may produce several inefficient small GEMMs plus communication.

### A practical optimization sequence

1. **Capture the real workload.** Record `M/N/K`, batch sizes, dtype, accumulation, strides, transpose flags, output fusion, and shape frequency. Benchmark warm executions and include packing or conversion when it occurs on every request.
2. **Establish a library baseline.** Compare cuBLAS/cuBLASLt with identical semantics. Specialization is useful when the actual workload has exploitable structure.
3. **Classify the limit.** Use arithmetic intensity and counters to distinguish bandwidth, compute, launch overhead, and inadequate parallelism.
4. **Improve data reuse.** Tile the output; reuse input tiles through shared memory and registers. Coalesce accesses and avoid bank conflicts or unnecessary layout conversions.
5. **Keep the arithmetic pipeline supplied.** Tune tile shapes, warps, pipeline stages, and register usage. Overlap supported asynchronous loads with computation. Larger tiles improve reuse but may reduce the number of resident blocks.
6. **Use the appropriate arithmetic.** Choose Tensor Core kernels and supported precision/layout combinations. Alignment and useful tile coverage matter; exact restrictions vary with hardware and libraries. Validate any precision change.
7. **Address small or irregular problems.** Consider grouped/batched GEMM; consider split-K when output dimensions provide too few tiles. Include the reduction and workspace cost of split-K.
8. **Fuse the epilogue when valuable.** Bias, activation, or residual fusion can save launches and intermediate memory traffic, but may increase registers and constrain kernel choice.

The CUDA guide discusses memory access and resource tradeoffs; CUTLASS exposes tiled GEMM building blocks and fused epilogues. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/overview.html).

For bandwidth-limited decode, batching can amortize weight reads, but it must be evaluated against per-request latency. Weight quantization saves bytes only when its dequantization and kernel costs do not erase the benefit.

**Interview takeaway:** first establish which shapes matter and why they are slow; then optimize the limiting resource and validate the end-to-end gain.

## 4. How do you locate and verify a performance bottleneck?

Start with a reproducible workload and a metric that represents the user or training objective. For serving, separate queueing, time to first token, time between tokens, and output-token throughput. Define the SLO precisely and report the rate of requests meeting it. For training, measure step time and useful tokens per second under comparable model quality conditions.

### Work from the system timeline to the expensive kernel

1. **Fix the experiment.** Record model revision, software, hardware, precision, parallel groups, input/output lengths, concurrency, and cache-hit rate. Warm up compilation and caches deliberately, and report whether the experiment is warm or cold.
2. **Split the critical path.** Add application/NVTX ranges for tokenization, queueing, prefill, decode, sampling, transfers, and collectives. Capture a representative window using Nsight Systems to locate gaps, synchronizations, or long dependencies. [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
3. **Remove one suspected cause.** Substitute pre-tokenized synthetic input; compare one GPU group against multiple groups; hold context length fixed while sweeping batch. Change one variable so the result can support or reject a hypothesis.
4. **Inspect dominant kernels.** Use Nsight Compute for the relevant instruction pipelines, memory hierarchy, stalls, spills, occupancy, and launch dimensions. Occupancy is supporting evidence, not a target to maximize independently. [Nsight Compute user guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
5. **Apply the smallest justified change.** State the expected counter and timing change before running it. For example: reducing repeated weight reads should lower bytes per token and improve decode time.
6. **Re-run the unprofiled application.** Check correctness, latency percentiles, throughput, and regressions across the workload distribution. Profiling overhead and replay can alter timings and concurrency.

A basic trace command, to adapt for an actual application, is:

```bash
nsys profile --trace=cuda,nvtx --output=timeline python workload.py
```

CUDA launches are asynchronous. A CPU timer around enqueueing a kernel measures submission rather than completion. For isolated device work, use CUDA events on the relevant stream, or synchronize the intended boundary. For end-to-end latency, use a wall-clock timer with well-defined completion. Per-operation synchronization inside the normal workload can destroy overlap and change the result.

### Interpret evidence, not a single utilization percentage

| Observation | Hypothesis to test | Useful follow-up |
| --- | --- | --- |
| GPU idle and no queue | Insufficient offered load or slow request generation | Increase client concurrency; bypass frontend stages |
| GPU idle with a long queue | CPU scheduling, synchronization, locks, or KV admission constraints | Inspect CPU/GPU timeline and scheduler decisions |
| High HBM throughput with modest arithmetic | Bandwidth-limited kernels | Estimate bytes per token and examine reuse |
| High arithmetic-pipeline activity near its applicable roof | Compute saturation | Compare precision, operation count, and an optimized baseline |
| Both bandwidth and arithmetic low | Small workloads, dependencies, or launch/latency limits | Inspect shapes, stalls, kernel gaps, and parallelism |
| Collectives dominate | Network limits, arrival skew, or poor overlap | Compare per-rank timelines and NCCL microbenchmarks |
| Average latency good, tail latency poor | Queue interference, stragglers, or memory pressure | Correlate slow requests with lengths, queues, and cache state |

Use CPU profiling for tokenization/host work, Nsight Systems for ordering and overlap, and Nsight Compute for kernel behavior. Avoid assuming that an apparent collective wait is caused by networking: another rank may arrive late.

**Interview takeaway:** provide a measured hypothesis, a controlled experiment, and an end-to-end result. A faster kernel outside the critical path may not make the application faster.

## 5. What is FlashAttention?

FlashAttention computes the same dense attention operation using an IO-aware execution order. It reduces transfers between HBM and faster on-chip storage instead of materializing the full score and probability matrices in HBM. It is algorithmically exact dense attention, although floating-point operation order can produce small numerical differences. [FlashAttention paper](https://arxiv.org/abs/2205.14135).

### Why ordinary attention moves so much data

```text
S = Q @ K.T / sqrt(head_dim)
P = softmax(S)
O = P @ V
```

For sequence length `L`, `S` and `P` each have `L*L` entries per head. At `L = 8192`, one two-byte score matrix alone occupies `128 MiB`. Writing it, reading it for softmax, and reading the probabilities again can create substantial traffic. Some implementations also use higher-precision temporaries.

### Online softmax allows processing by tiles

Process one query tile against successive key/value tiles. For each query row, maintain a running maximum `m`, denominator `l`, and unnormalized weighted-value accumulator `u`. Given a new score block `s` and its value tile `V_block`:

```text
Initially: m = -infinity, l = 0, u = 0

m_new = max(m, max(s))
alpha = exp(m - m_new)
p     = exp(s - m_new)

l_new = alpha * l + sum(p)
u_new = alpha * u + p @ V_block

After all key/value blocks: output = u / l
```

Rescaling the earlier accumulators when the maximum changes keeps all blocks normalized consistently. A causal implementation skips or masks future scores. Assume each row has an allowed key; implementations need special handling for completely masked blocks or rows. The recurrence avoids storing full attention matrices. Backward can recompute local values from inputs and saved normalization statistics instead of saving all probabilities. [FlashAttention algorithm](https://arxiv.org/pdf/2205.14135).

Standard dense attention still performs quadratic work in sequence length. FlashAttention reduces intermediate storage and IO; it does not turn dense attention into linear-time attention. Its temporary-storage savings also do not remove the persistent autoregressive KV cache.

### When it helps and what remains to measure

Long-sequence prefill often benefits because large attention intermediates are expensive. A one-token decode step has a different shape: no large square score matrix exists, while reading historical KV and launching enough parallel work can dominate. Benchmark the relevant decode kernel separately rather than assuming prefill speedups transfer unchanged.

Tiling is only part of performance. FlashAttention-2 improves work partitioning and parallelism and reduces some non-matmul work; the fastest implementation depends on hardware and shape. [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691).

**Interview takeaway:** explain the avoided memory traffic and online normalization, then distinguish mathematical complexity, temporary attention storage, and persistent KV storage.

## 6. How do you debug a C++ out-of-bounds write using a core dump?

A core dump captures a process snapshot, not its execution history. It can show corrupted memory and the eventual crash, but it may not identify the instruction that wrote outside the array earlier. A crash in `free()`, a container, or a virtual call may be a consequence rather than the source of corruption.

### Inspect the crash with the exact artifacts

Preserve the executable, matching debug symbols, shared libraries, build ID, input, configuration, and core. Debugging against a newly rebuilt executable can produce misleading frames and object layouts. Some memory mappings may be absent from a core. [GDB files and core-file handling](https://www.sourceware.org/gdb/current/onlinedocs/gdb.html/Files.html).

On Linux, an initial investigation might use:

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

Identify the signal, failing access, and affected object. Inspect pointer values, lengths, allocation boundaries, and nearby bytes; use `p`, `x`, and disassembly when symbols or optimized variables are insufficient. Check all threads for writers and ownership changes, while treating badly corrupted backtraces cautiously.

### Work backward from the victim to plausible writers

Suppose a pointer-sized field immediately following a fixed array contains a recognizable repeated value. That suggests a possible overrun, but allocator layout and object placement must be verified rather than assumed. Search code that can reach the affected allocation for:

- A loop using `i <= size` instead of `i < size`.
- `memcpy` lengths that confuse elements with bytes.
- Incorrect tensor strides, shape products, or signed/unsigned conversions.
- Integer overflow in allocation-size calculations.
- Concurrent resizing, dangling pointers, or use-after-free that resembles an overrun.

Compare the damaged byte range with the candidate writes and input sizes. The location of a corrupted structure alone is not proof of the guilty function.

### Reproduce and stop at the first invalid write

Build a reproducer with debug information and instrumentation:

```bash
clang++ -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer repro.cpp -o repro_asan
./repro_asan
```

AddressSanitizer detects many heap/stack/global bounds errors and use-after-free cases and reports the invalid access and relevant allocation history. Instrument the affected components, not only an unrelated wrapper. It cannot reconstruct an old unsanitized core, and a write within one allocation that corrupts another field may evade ordinary bounds detection. [AddressSanitizer documentation](https://clang.llvm.org/docs/AddressSanitizer.html).

Alternatively, on a new live run, stop after the victim is allocated and set a hardware watchpoint on a small affected field:

```gdb
break initialize_victim
run
# Advance to a point where victim exists and has its expected value.
watch -location victim->field
continue
```

Recompute the address on that run; ASLR and allocation changes make an address from the old core unreliable. Watchpoints have hardware size/count limits, and normal writes may trigger before the unwanted one. They require execution; a frozen core cannot trigger a new watchpoint. [GDB watchpoints](https://www.sourceware.org/gdb/current/onlinedocs/gdb.html/Set-Watchpoints.html).

Finally, fix the bounds or ownership error and add a reproducer covering the exact boundary and nearby sizes. If no reproduction or execution recording exists, clearly separate what the core proves from the suspected earlier writer.

**Interview takeaway:** use the core to establish the damaged state, then use instrumentation or a live watchpoint to catch the cause when possible.

## 7. Implement an attention module with Grouped Query Attention (GQA)

### Head sharing and the memory benefit

GQA gives several query heads a shared key head and value head. Let `Hq` be the number of query heads and `Hkv` the number of KV heads, requiring `Hq % Hkv == 0`:

```text
group_size = Hq / Hkv
kv_head_for_query_head(h) = floor(h / group_size)

Hkv = Hq: ordinary multi-head attention
Hkv = 1:  multi-query attention
1 < Hkv < Hq: grouped-query attention
```

The Q projection has `Hq * Dh` output features; K and V each have `Hkv * Dh`. Query heads still have separate attention distributions, even when they share K/V. [GQA paper](https://arxiv.org/abs/2305.13245).

For example, with eight query heads and two KV heads:

```text
Query heads 0, 1, 2, 3 -> KV head 0
Query heads 4, 5, 6, 7 -> KV head 1
```

For `layers` decoder layers and `live_tokens` cached positions:

```text
KV bytes = 2 * layers * live_tokens * Hkv * Dh * bytes_per_element
```

Relative to MHA with the same `Hq`, head size, and KV precision, persistent KV storage is reduced by `Hq/Hkv`. Total model memory and total computation do not shrink by that same factor. GQA trades the expressiveness of independently learned KV heads against memory and decoding efficiency; quality depends on model training and configuration.

### Complete NumPy implementation

This example implements causal self-attention with optional KV caching. It omits RoPE, dropout, padding masks, and training gradients so the grouping is visible. It uses float64 for a numerical reference; a production kernel would use an appropriate device dtype and fused execution.

```text
x:       (B, S, D)
q:       (B, Hkv, group_size, S, Dh)
k/v:     (B, Hkv, P + S, Dh)
scores:  (B, Hkv, group_size, S, P + S)
output:  (B, S, D)
```

The singleton group dimension of K/V is broadcast across query heads in that group. The stored KV cache retains only `Hkv` heads; the code does not repeat it into `Hq` heads.

```python
import numpy as np


class GQAAttention:
    def __init__(self, d_model, num_query_heads, num_kv_heads, seed=0):
        if min(d_model, num_query_heads, num_kv_heads) <= 0:
            raise ValueError("Dimensions and head counts must be positive")
        if d_model % num_query_heads or num_query_heads % num_kv_heads:
            raise ValueError("Require D % Hq == 0 and Hq % Hkv == 0")
        self.d_model = d_model
        self.hq = num_query_heads
        self.hkv = num_kv_heads
        self.dh = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, self.hkv * self.dh))
        self.Wv = rng.normal(0, scale, (d_model, self.hkv * self.dh))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def forward(self, x, cache=None):
        """Return new-token outputs and compact projected (K, V) cache."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != self.d_model or x.shape[1] == 0:
            raise ValueError("Expected nonempty input with shape (B, S, D)")
        batch, length, _ = x.shape
        q = (x @ self.Wq).reshape(
            batch, length, self.hkv, self.group_size, self.dh
        ).transpose(0, 2, 3, 1, 4)
        k = (x @ self.Wk).reshape(
            batch, length, self.hkv, self.dh
        ).transpose(0, 2, 1, 3)
        v = (x @ self.Wv).reshape(
            batch, length, self.hkv, self.dh
        ).transpose(0, 2, 1, 3)

        past_length = 0
        if cache is not None:
            past_k, past_v = cache
            if (past_k.ndim != 4 or past_k.shape != past_v.shape
                    or past_k.shape[:2] != (batch, self.hkv)
                    or past_k.shape[3] != self.dh):
                raise ValueError("Cache must have shape (B, Hkv, P, Dh)")
            past_length = past_k.shape[2]
            k = np.concatenate((past_k, k), axis=2)
            v = np.concatenate((past_v, v), axis=2)

        scores = q @ k[:, :, None, :, :].swapaxes(-1, -2)
        scores /= np.sqrt(self.dh)
        key_positions = np.arange(k.shape[2])[None, :]
        query_positions = past_length + np.arange(length)[:, None]
        future = key_positions > query_positions
        scores = np.where(future[None, None, None, :, :], -np.inf, scores)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        context = probabilities @ v[:, :, None, :, :]
        merged = context.transpose(0, 3, 1, 2, 4).reshape(
            batch, length, self.d_model
        )
        return merged @ self.Wo, (k, v)
```

Each local query position `i` is globally at `past_length + i`. The mask therefore permits keys through that position, including all historical keys and the current token. A plain square triangular mask would be insufficient for multi-token calls with an existing cache.

### Executable numerical checks

Run this block after the implementation above. The oracle uses explicit K/V-head repetition and per-query loops rather than the implementation's grouped broadcasting. Repetition is acceptable in a small correctness test; it would waste memory in the actual cache.

```python
def reference_full_attention(model, x):
    batch, length, dimension = x.shape
    q = (x @ model.Wq).reshape(
        batch, length, model.hq, model.dh
    ).transpose(0, 2, 1, 3)
    k = (x @ model.Wk).reshape(
        batch, length, model.hkv, model.dh
    ).transpose(0, 2, 1, 3)
    v = (x @ model.Wv).reshape(
        batch, length, model.hkv, model.dh
    ).transpose(0, 2, 1, 3)
    k = np.repeat(k, model.group_size, axis=1)
    v = np.repeat(v, model.group_size, axis=1)
    context = np.empty_like(q)
    for b in range(batch):
        for h in range(model.hq):
            for t in range(length):
                logits = k[b, h, :t + 1] @ q[b, h, t] / np.sqrt(model.dh)
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                context[b, h, t] = weights @ v[b, h, :t + 1]
    merged = context.transpose(0, 2, 1, 3).reshape(batch, length, dimension)
    return merged @ model.Wo


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


def run_checks():
    cases = 0
    for batch, length in ((1, 1), (2, 7), (3, 5)):
        for hq, hkv in ((8, 8), (8, 2), (8, 1)):
            model = GQAAttention(32, hq, hkv, seed=11)
            x = np.random.default_rng(13).normal(size=(batch, length, 32))
            original = x.copy()
            expected = reference_full_attention(model, x)
            full, full_cache = model.forward(x)
            check_close(full, expected)
            assert full_cache[0].shape == (batch, hkv, length, model.dh)
            assert full_cache[1].shape == (batch, hkv, length, model.dh)

            cache, pieces = None, []
            for t in range(length):
                part, cache = model.forward(x[:, t:t + 1], cache)
                pieces.append(part)
                assert cache[0].shape[2] == t + 1
            check_close(np.concatenate(pieces, axis=1), expected)
            check_close(cache[0], full_cache[0])
            check_close(cache[1], full_cache[1])

            if length > 2:
                prefix, prefix_cache = model.forward(x[:, :2])
                saved_cache = tuple(item.copy() for item in prefix_cache)
                suffix, chunk_cache = model.forward(x[:, 2:], prefix_cache)
                check_close(np.concatenate((prefix, suffix), axis=1), expected)
                for old, saved, final, full_item in zip(
                    prefix_cache, saved_cache, chunk_cache, full_cache
                ):
                    check_close(old, saved)
                    check_close(final, full_item)
                changed = x.copy()
                changed[:, 2:] += 100.0
                changed_output, _ = model.forward(changed)
                check_close(changed_output[:, :2], full[:, :2])

            for b in range(batch):
                single, _ = model.forward(x[b:b + 1])
                check_close(single[0], full[b])
            np.testing.assert_array_equal(x, original)
            cases += 1

    print(f"Passed {cases} MHA/GQA/MQA configurations and all cache checks.")


run_checks()
```

Expected output:

```text
Passed 9 MHA/GQA/MQA configurations and all cache checks.
```

For production, add the model's positional encoding before caching the appropriate keys, padding/ragged-sequence handling, numerical validation for the target dtype, and an optimized grouped-attention kernel. The explicit score tensor above is a readable reference, not a FlashAttention implementation.

**Interview takeaway:** project fewer K/V heads, map each query head to the correct KV group, retain per-query-head softmax, and preserve compact K/V storage during decoding.
