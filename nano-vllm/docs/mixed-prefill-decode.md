# Mixed Prefill/Decode Scheduling

## 背景

nano-vllm 的调度器目前每一步要么全 prefill、要么全 decode。当一个大 prefill 运行时，所有 decode 序列必须等待整个 prefill-only step 完成才能继续，导致 decode 尾延迟（tail latency）升高。

目标：让 prefill 和 decode 能在**同一次**前向传播中混跑，降低 decode 延迟。核心思路是利用 `flash_attn_varlen_func` + `block_table` 原生支持混合长度序列的特性——decode 序列就是 `seqlen_q=1` 的 prefill。

## 当前架构分析

### Scheduler (`nanovllm/engine/scheduler.py`)

```
schedule():
  Phase 1 — prefill（从 waiting heap 取）
    如果拿到 seq → return (seqs, True)   ← 全 prefill step
  Phase 2 — decode（从 running deque 取）
    只有 Phase 1 没拿到才执行 → return (seqs, False)  ← 全 decode step
```

关键约束：
- `is_prefill` 是 step 级别的 bool，不是 per-sequence
- `postprocess(seqs, token_ids, is_prefill)` 依赖这个 bool 来判断是否跳过 mid-prefill 的 token 生成

### ModelRunner (`nanovllm/engine/model_runner.py`)

```
run(seqs, is_prefill):
  if is_prefill:  prepare_prefill(seqs)   → varlen tensors
  else:           prepare_decode(seqs)    → 1-token-per-seq tensors
  run_model(is_prefill) → CUDA graph 仅用于 decode
```

### Attention (`nanovllm/layers/attention.py`)

```python
if context.is_prefill:
    if context.block_tables is not None:
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(q, k, v, cu_seqlens_q/k, block_table=...)
else:
    o = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=..., block_table=...)
```

两种 attention kernel：
- **Prefill**: `flash_attn_varlen_func` — varlen，Q/K/V 是当前层输出（或 cache）
- **Decode**: `flash_attn_with_kvcache` — 每个 seq 只有 1 个 query token，K/V 全从 cache 读

### 为什么 prefill 和 decode 不能混跑（当前限制）

1. Scheduler 的 `schedule()` 是全有或全无：拿到 prefill 就立即返回，decode 只在 prefill 为空时才执行
2. `prepare_prefill` 和 `prepare_decode` 构建完全不同的 tensor 布局（varlen vs 1-token-per-seq）
3. Attention kernel 依赖 `context.is_prefill` 做二选一分发
4. CUDA graph 只捕获了 decode-only 的固定 shape

## 设计方案

### 核心思路：统一 Attention 路径

`flash_attn_varlen_func` + `block_table` 可以原生处理混合长度序列。decode 序列就是 `seqlen_q=1` 的 prefill：

```
所有序列 →
  QKV projection → q, k, v（prefill 多 token + decode 单 token）
  store_kvcache → 写入 cache
  flash_attn_varlen_func(q, k_cache, v_cache,
                         cu_seqlens_q, cu_seqlens_k,
                         block_table=...)  ← 统一路径
```

`cu_seqlens_q` 和 `cu_seqlens_k` 描述每个序列的 query/KV 长度：
- Prefill 序列：`seqlen_q = num_scheduled_tokens`, `seqlen_k = num_cached_tokens + num_scheduled_tokens`
- Decode 序列：`seqlen_q = 1`, `seqlen_k = num_tokens`

`ParallelLMHead` 无需修改：`last_indices = cu_seqlens_q[1:] - 1` 对 prefill（最后一个 prefill token）和 decode（唯一的 token）都正确。

### Scheduler：一步两阶段

```
schedule():
  Phase 1 — Prefill（从 waiting heap 取，逻辑不变）
    尊重 max_num_batched_tokens 和 max_num_seqs
    分配 KV blocks，设置 num_scheduled_tokens
    
  Phase 2 — Decode（从 running deque 取，填满剩余配额）
    条件：len(seqs) < max_num_seqs AND num_tokens < max_num_batched_tokens
    每个 decode 消耗 1 token budget（num_batched_tokens += 1）
    
  返回 (seqs, has_prefill)   ← has_prefill = any(s.is_prefill for s in seqs)
```

`postprocess` 不再接收 `is_prefill` 参数，改为 per-sequence 判断：

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue  # mid-prefill skip
        seq.append_token(token_id)
        # ... 原有的 finish/cleanup 逻辑
```

### ModelRunner：prepare_mixed()

新增统一函数构建 varlen tensors：

```python
def prepare_mixed(self, seqs):
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens    # prefill: N, decode: 1
        end = start + seqlen_q

        if seq.is_prefill:
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
        else:
            input_ids.append(seq.last_token)
            positions.append(seq.num_tokens - 1)

        seqlen_k = end  # = num_cached_tokens + num_scheduled_tokens
        cu_seqlens_q.append(prev + seqlen_q)
        cu_seqlens_k.append(prev + seqlen_k)

        # slot_mapping 逻辑对 prefill 和 decode 完全相同
        if seq.block_table:
            for i in range(start_blk, end_blk):
                ...
                slot_mapping.extend(...)

    set_context(True, cu_seqlens_q, cu_seqlens_k, ...,
                slot_mapping, None, block_tables)
```

关键点：
- `seq.is_prefill` 分支仅决定输入来源（prompt tokens vs `last_token`）和 positions
- `seqlen_k = end` 对 prefill 和 decode 都正确
- `if seq.block_table:` 跳过 warmup（无 cache 的场景）
- `block_tables` 在 `any(seq.block_table)` 且 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 时设置
- 设置 `is_prefill=True` → attention 走 varlen 路径

### CUDA Graph

- **纯 decode batch**（`has_prefill=False`）：完整 CUDA graph replay，无回归
- **混合 batch**（`has_prefill=True`）：eager 模式（token 数量可变，无法 graph）
- Partial CUDA Graph（仅 MLP 层）延后到 Phase 2

## 实现步骤

### Step 1：改写 Scheduler

**文件**：`nanovllm/engine/scheduler.py`

- Phase 1（prefill）逻辑不变
- Phase 2（decode）现在在 Phase 1 有结果时也执行
- Phase 2 while 条件增加 `num_batched_tokens < self.max_num_batched_tokens`
- Phase 2 增加 `num_batched_tokens += 1`
- 仅 Phase 2 的 decode 序列 `extendleft` 回 running（Phase 1 完成 prefill 的序列已通过 `.append()` 加入 running）
- 返回 `(scheduled_seqs, has_prefill)`

### Step 2：改写 postprocess

**文件**：`nanovllm/engine/scheduler.py`

- 移除 `is_prefill` 参数
- per-sequence 判断 mid-prefill skip：`if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens: continue`

### Step 3：新增 prepare_mixed()

**文件**：`nanovllm/engine/model_runner.py`

- 按上述 spec 实现
- warmup 保持用现有的 `prepare_prefill`（无 block_tables，纯 prefill）

### Step 4：修改 run()

**文件**：`nanovllm/engine/model_runner.py`

- 签名：`run(self, seqs, has_prefill)` 替代 `run(self, seqs, is_prefill)`
- `has_prefill=True` → `prepare_mixed(seqs)` + eager forward
- `has_prefill=False` → `prepare_decode(seqs)` + CUDA graph replay

### Step 5：更新 LLMEngine

**文件**：`nanovllm/engine/llm_engine.py`

- 接收 `has_prefill` 并传给 model_runner
- `postprocess()` 调用不再传 `is_prefill`

### Step 6：验证 Attention

**文件**：`nanovllm/layers/attention.py`

现有代码已正确处理——混合 batch 设置 `is_prefill=True` + `block_tables` → 走 varlen 路径。无需逻辑改动，仅添加注释说明混合 batch 语义。

### Step 7：测试

#### 单元测试 — `tests/test_scheduler_mixed.py`（新建）

沿用 `test_scheduler_preempt.py` 的模式：`Config.__new__` + `Scheduler.__new__` + `BS=4` + 手动构造 sequence/block 状态。

| 测试 | 场景 |
|------|------|
| `test_mixed_batch_basic` | 1 waiting + 1 running → 同一步调度 |
| `test_token_budget_split` | 大 prefill + 多个 decode，验证预算分配 |
| `test_pure_decode_fallback` | waiting 为空 → 退化为纯 decode |
| `test_pure_prefill_fallback` | running 为空 → 退化为纯 prefill |
| `test_chunked_prefill_with_decode` | 大 prefill 跨多步 chunk，每步混入 decode |
| `test_preempt_in_mixed` | 混合模式 block 不足 → victim eviction 正确 |
| `test_self_preempt_mixed` | 单 running seq 在 block 边界无空闲 block → 自抢占 |
| `test_budget_exhaustion_by_prefill` | 小 budget，Phase 1 耗尽 → Phase 2 不执行 |
| `test_mixed_prefill_completion` | prefill 完成 + decode 完成在同一步 → postprocess 正确 |

#### 端到端测试

适配 `test_preempt_correctness.py`：对比纯 decode 调度与混合调度的输出一致性（低 `gpu_memory_utilization` 强制触发混合 batch）。

#### TTFT / Decode Latency 测试 — `tests/test_ttft.py`

##### 动机

混合调度的核心收益是降低 decode 尾延迟。在 all-or-nothing 调度下，decode 序列必须等 prefill-only step 完整执行完才能拿到下一个 token，导致 inter-token latency (ITL) 出现尖峰。

测试已实现，可直接运行基线版本：

```bash
python tests/test_ttft.py                    # Qwen3 0.6B (默认)
python tests/test_ttft.py -v                 # verbose: 显示每步调度组成
python tests/test_ttft.py --model llama      # LLaMA 3.1 8B
```

##### 测试场景

提交 N 个短请求（2 tokens prompt, `max_tokens=20`）+ M 个长请求（~2000 tokens prompt, chunked prefill）。短请求快速完成 prefill 进入 decode 队列，长请求触发多步 prefill 阻塞短请求的 decode。

在 all-or-nothing 调度下（当前 baseline）：
```
step 1:  10 short prefills (20 tok)          → short 完成 prefill, 生成 token 0
step 2-5: long prefill chunks (512×3+415)    → short 在 running 队列中被阻塞!
step 6-23: decode (11 tok/step)              → short 生成 token 1-18
step 24:  decode (11 tok)                    → short 生成 token 19, 所有请求完成
```

##### 实现方式

- **Monkey-patch `Scheduler.postprocess`** 记录每个 token 生成的时间戳
- **`max_tokens > 1`** 让短请求进入 decode 阶段（prefill 的 postprocess 生成 token 0 后继续在 running 队列中）
- 追踪每个请求的 **ITL (Inter-Token Latency)**：连续 token 之间的时间间隔

##### 指标

| 指标 | 含义 | Baseline 预期 | Mixed 预期 |
|------|------|---------------|------------|
| **ITL p50** | 中位 token 间隔 | ~28ms (正常 decode) | ~28ms |
| **ITL p99** | 99分位 token 间隔 | ~250ms (被 prefill 阻塞) | ~28ms |
| **Completion** | 请求完成时间 | ~2s | 略低（少等 4 步） |
| **Total steps** | 总调度步数 | 24 | 20（少 4 步纯 prefill） |

核心判据：
- **PASS**: `mixed ITL p99 <= baseline ITL p99 * 0.5`（尾延迟显著降低）
- **PASS**: `mixed ITL p50 ≈ baseline ITL p50`（正常 decode 不退化）

##### Baseline 实测结果 (Qwen3-0.6B)

```
Total time: 2.10s  |  Steps: 24
Short requests (n=10):
  TTFT:       p50=1321ms   p99=1321ms     ← prefill postprocess 生成
  ITL:        p50=28.9ms   p99=255.9ms    ← 被 4 个 prefill step 阻塞
  Completion: p50=2104ms
Long requests (n=1):
  ITL:        p50=28.9ms   p99=36.1ms     ← 无阻塞（prefill 完成后才 decode）
```

ITL 分布的显著差异（short p99=256ms vs long p99=36ms）正是 all-or-nothing 调度造成的 decode 阻塞。混合调度应消除这个差异。

## 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `nanovllm/engine/scheduler.py` | 修改 | 两阶段 `schedule()`，`postprocess()` 移除 `is_prefill` 参数 |
| `nanovllm/engine/model_runner.py` | 修改 | 新增 `prepare_mixed()`，修改 `run()` 签名 |
| `nanovllm/engine/llm_engine.py` | 修改 | 适配新的 `schedule()` 返回值和 `postprocess()` 签名 |
| `nanovllm/layers/attention.py` | 微调 | 添加注释（无逻辑改动） |
| `tests/test_scheduler_mixed.py` | 新建 | 混合调度器单元测试（9 个） |
| `tests/test_ttft.py` | 新建 | TTFT 对比 benchmark |
| `docs/mixed-prefill-decode.md` | 新建 | 本文档 |

**不修改**：`models/`、`layers/linear.py`、`layers/sampler.py`、`layers/embed_head.py`、`block_manager.py`、`sequence.py`、`config.py`、`context.py`

## 验证计划

1. **已有测试无回归**：
   ```bash
   python tests/test_scheduler_preempt.py
   python tests/test_preempt_correctness.py
   ```

2. **新单元测试**：
   ```bash
   python tests/test_scheduler_mixed.py
   ```

3. **TTFT 对比测试** — 验证混合调度的尾延迟改善：
   ```bash
   python tests/test_ttft.py --model qwen3 --mode compare
   ```

4. **benchmark** — 确认吞吐无回归（纯 decode 仍走 CUDA graph）：
   ```bash
   python bench_qwen3.py
   python bench_llama.py
   ```

4. **冒烟测试** — 生成文本质量不变：
   ```bash
   python example_qwen3.py
   python example_llama.py
   ```
