# Mixed Prefill/Decode Scheduling

## 背景

nano-vllm 的调度器每一步要么全 prefill、要么全 decode。当一个大 prefill 运行时，所有 decode 序列必须等待整个 prefill-only step 完成才能继续，导致 decode 尾延迟（tail latency）升高。

目标：让 prefill 和 decode 能在**同一次**前向传播中混跑，降低 decode 延迟。核心思路是利用 `flash_attn_varlen_func` + `block_table` 原生支持混合长度序列的特性——decode 序列就是 `seqlen_q=1` 的 prefill。

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

新增统一函数构建 varlen tensors。关键点：

- `seq.is_prefill` 分支仅决定输入来源（prompt tokens vs `last_token`）和 positions
- `seqlen_k = end` 对 prefill 和 decode 都正确
- `if seq.block_table:` 跳过 warmup（无 cache 的场景）
- `block_tables` 在 `any(seq.block_table)` 且 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 时设置
- 设置 `is_prefill=True` → attention 走 varlen 路径

### CUDA Graph

- **纯 decode batch**（`has_prefill=False`）：完整 CUDA graph replay，无回归
- **混合 batch**（`has_prefill=True`）：eager 模式（token 数量可变，无法 graph）

## 实现过程与遇到的问题

### 问题 1：Dual-Scheduling Bug（同一 seq 被调度两次）

**现象**：`prepare_mixed` 在 Phase 2 对某 seq 提取 `seq.last_token` 时，发现该 seq 刚在 Phase 1 完成 prefill、`last_token` 还是初始值，导致尝试提取越界 token。

**根因**：Phase 1 完成 prefill 后直接把 seq 加入 `self.running`，Phase 2 从 `self.running` 弹出同一个 seq，覆盖了 Phase 1 设置的 `is_prefill=True` 和 `num_scheduled_tokens`，然后试图作为 decode 序列处理。

**修复**：引入 `fresh_running` 列表。Phase 1 完成 prefill 的 seq 暂存到 `fresh_running`，Phase 2 结束后才追加到 `self.running`。Phase 2 只处理 Phase 1 之前已在 `self.running` 中的 seq。

```python
# Phase 1
fresh_running = []
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    heapq.heappop(self.waiting)
    fresh_running.append(seq)       # 暂存，不直接加入 running

# Phase 2（只处理 self.running 中的旧 seq）
...

# Phase 2 之后
for seq in fresh_running:
    self.running.append(seq)        # 此时再加入
```

### 问题 2：TTFT Monkey-Patch 签名不匹配

**现象**：`test_ttft.py` 报 `ValueError: too many values to unpack`。

**根因**：`llm.step()` 改为返回 3 个值 `(outputs, num_tokens, has_prefill)`，但 TTFT 测试仍用 2 元组解包。此外，monkey-patch 的 `timed_postprocess` 仍接受旧的 `is_prefill` 参数。

**修复**：
1. `output, num_tokens, has_prefill = llm.step()`
2. `timed_postprocess` 移除 `is_prefill` 参数，改为读取 `seq.is_prefill`

### 问题 3：Straggler 1-Token Batch

**现象**：TTFT 基准测试显示最终 steps 出现只有 1 个 token 的 batch，单步耗时 ~700ms（对比正常 11-token batch 的 ~28ms）。这是 CUDA kernel launch overhead 在极小 batch 下的表现。

**根因分析**：若干短请求在同一 step 完成（`num_completion_tokens == max_tokens - 1`），最后一个 step 只剩 1 条长请求还在 decode，形成 1-token batch。

**修复：Straggler Deferral 优化**

在 Phase 2 收集 `decode_candidates` 后，检查：如果调度完所有候选后 `running` 只剩 1 条非即将完成的 seq，且 waiting 为空（没有新的 prefill 可混合），则推迟部分即将完成的 seq，保留 1 条来"陪伴"末端的 straggler：

```python
if decode_candidates:
    finishing = [s for s in decode_candidates
                 if s.num_completion_tokens == s.max_tokens - 1]
    non_finishing = len(decode_candidates) - len(finishing)
    running_after = len(self.running) + len(fresh_running) + non_finishing
    if running_after == 1 and not self.waiting and finishing and non_finishing > 0:
        defer = finishing[1:]  # keep 1 to pair with the straggler
        for s in reversed(defer):
            self.running.appendleft(s)
        decode_candidates = [s for s in decode_candidates if s not in defer]
```

第一次尝试推迟全部 finishing seq，结果将 1-token batch 从 step 24 移到了 step 23，问题未解决。**关键修正**：只推迟 `finishing[1:]`（保留 1 条），让 step 23 有 2 token、step 24 有剩余 token，消除了所有 1-token batch。

### 问题 4：预算分配——Phase 1 耗尽全部 budget

**现象**：TTFT 基准测试显示 mixed 调度几乎没有效果——Phase 1 的大 prefill 消耗了全部 `max_num_batched_tokens`，Phase 2 拿不到任何 token budget，decode 仍然被阻塞。straggler deferral 只是治标。

**用户指出根因**：Phase 1 没有为 running 中的 decode seq 预留配额。

**修复：预算预留（Budget Reservation）**

在 Phase 1 循环前计算 `decode_reserve = len(self.running)`，Phase 1 的 `remaining` 计算减去这个预留：

```python
# Phase 1 之前
decode_reserve = len(self.running) if self.running else 0

# Phase 1 循环内
remaining = self.max_num_batched_tokens - num_batched_tokens - decode_reserve
if remaining <= 0:     # 注意：原来是 == 0，现在改为 <= 0
    break
```

这样 Phase 1 最多使用 `max_batched - len(running)` 个 token，Phase 2 的每个 running seq 都有 1 token 的预算来跑 decode，mixed P/D 真正生效。

### 问题 5：TOCTOU —— Phase 2 的 Block 分配竞态

**现象**：`bench.py`（256 条序列，`ignore_eos=True`）报 `IndexError: pop from an empty deque`，崩溃在 `_allocate_block` → `popleft()`。

**根因**：Phase 2 的 while 循环中对每个 decode seq 调用 `can_append()` 检查是否有空闲 block，但 `may_append()` 在之后的 for 循环才执行实际分配。多个候选 seq 在检查时看到同一个空闲 block，分配时第二个就拿不到了（TOCTOU = Time-Of-Check-Time-Of-Use）。

在纯 decode 调度下这个问题不常见，因为 Phase 1 不消耗 block。但 mixed 调度下 Phase 1 可能消耗大量 block，Phase 2 的 block 余量紧张，概率性触发。

**修复：分配前二次检查**

在 `may_append()` 前再次调用 `can_append()`，失败则把 seq 放回 `running` 队首，下个 step 重试：

```python
for seq in decode_candidates:
    if not self.block_manager.can_append(seq):
        self.running.appendleft(seq)     # 放回队首，下步重试
        continue
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
    num_batched_tokens += 1
```

## 调试与排查方法

### 单元测试驱动

`tests/test_scheduler_mixed.py` 采用轻量级 mock 方式，不加载模型：
- `Config.__new__(Config)` 绕过正常初始化
- `Scheduler.__new__(Scheduler)` 手动注入 `block_manager`、`waiting`、`running`
- `BS=4` 固定 block size，手动构造 sequence 和 block 状态
- 每个测试独立设置 `num_blocks`、`max_batched`、`max_seqs`

这种方式使每个测试在 ~1ms 内完成，可以快速迭代验证调度逻辑。

### TTFT Benchmark 作为系统级探针

`test_ttft.py` monkey-patch `postprocess` 记录每个 token 的时间戳，精确测量：
- **TTFT**（Time To First Token）：从请求提交到第一个 token 的时间
- **ITL**（Inter-Token Latency）：连续 token 之间的间隔，反映 decode 是否被阻塞
- Per-step 调度组成（`--verbose`）：每步是 mixed/pure-decode，多少 token

这些指标直接暴露调度问题：ITL p99 尖峰 = decode 被阻塞，1-token batch = straggler 问题。

### 对比验证

`test_preempt_correctness.py` 用相同的 prompt 在 baseline（大 KV cache，无 preemption）和 tight（小 KV cache，触发 preemption+混合调度）下各跑一次，对比输出 token 序列是否一致。这是正确性验证的 gold standard。

## 最终测试结果

### 单元测试

```
test_scheduler_preempt.py    8/8 PASS
test_scheduler_mixed.py      9/9 PASS
  - test_mixed_batch_basic           1 waiting + 1 running → 同一步调度
  - test_token_budget_split          预算预留后 prefill=11, decode=1
  - test_pure_decode_fallback        waiting 为空 → pure decode
  - test_pure_prefill_fallback       running 为空 → pure prefill
  - test_chunked_prefill_with_decode 30-token prefill 分 4 步(9+9+9+3)，每步混入 decode
  - test_budget_exhaustion_by_prefill max_seqs=1 → prefill 耗尽 seq 槽位
  - test_mixed_prefill_completion    prefill 完成 + decode 同一步 → postprocess 正确
  - test_fresh_seq_not_in_phase2     新完成的 prefill 不会在同一步被当 decode 调度
  - test_mid_prefill_not_appended_to_running   chunked 中段保持在 waiting
```

### TTFT 基准测试（Qwen3-0.6B）

配置：10 short (2 tok) + 1 long (~1951 tok), `max_batched=512`, `max_tokens=20`

```
总时间: 1.98s / 24 steps

Step  1: mixed(20 tok)    ← 10 short prefill (2 tok each)
Step  2: mixed(512 tok)   ← long prefill chunk + decode
Step  3: mixed(512 tok)
Step  4: mixed(512 tok)
Step  5: mixed(455 tok)   ← long prefill completes
Step 6-24: decode(11 tok) ← 11 seqs decoding together

Short ITL: p50=28.6ms  p95=35.4ms  p99=143.7ms
Long ITL:  p50=28.5ms  p95=31.2ms  p99=34.6ms
```

零 1-token batch，step 24 是 `decode(7 tok)`，7 条一起完成。

### TTFT 基准测试（Llama 3.1 8B）

```
总时间: 2.28s / 24 steps

Short ITL: p50=28.8ms  p95=55.5ms  p99=137.7ms
Long ITL:  p50=27.9ms  p95=31.2ms  p99=33.4ms
```

### 吞吐基准

| Benchmark | 结果 |
|-----------|------|
| `bench.py` (Qwen3 0.6B, 256 seqs) | 133,966 tok / 25.14s = **5,328 tok/s** |
| `bench_llama.py` (Llama 3.1 8B, 64 seqs) | 11,269 tok / 12.04s = **936 tok/s** |

### 冒烟测试

`example.py` 和 `example_llama.py` 均正常生成高质量文本，无退化。

## 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `nanovllm/engine/scheduler.py` | 修改 | 两阶段 `schedule()` + budget reservation + straggler deferral + TOCTOU 修复 |
| `nanovllm/engine/model_runner.py` | 修改 | 新增 `prepare_mixed()`，`run()` 用 `has_prefill` 分发 |
| `nanovllm/engine/llm_engine.py` | 修改 | `step()` 返回 3 值，`postprocess()` 无 `is_prefill` 参数 |
| `tests/test_scheduler_mixed.py` | 新建 | 9 个混合调度器单元测试 |
| `tests/test_ttft.py` | 修改 | 适配新 API，monkey-patch 使用 `seq.is_prefill` |
| `docs/mixed-prefill-decode.md` | 更新 | 本文档 |
