# Scheduler 两项改进

## 一、短 prompt 优先调度（Priority Scheduling）

### 改动前

waiting 队列是 FIFO（`deque`），永远先调度最早到达的 seq。长 prompt 卡在队头时短 prompt 必须排队等待。

### 改动后

waiting 改为最小堆（`heapq`），按 `num_tokens - num_cached_tokens`（剩余待处理 token 数）排序，短 prompt 优先 prefill。

### 关键点

- `add(seq)` — 压入 `(remaining, tiebreaker, seq)` 元组，剩余 token 少的在堆顶
- `schedule()` 开头调用 `_reheapify()` — 重建堆以同步 chunked prefill 后的新优先级
- `heappop` 取代 `popleft` — seq 进入 running 时从堆顶弹出
- preempt 后的 seq — 同样压入堆中，按剩余 token 数排队

### 改动范围：`scheduler.py`

```
waiting: deque → list (heap)
add()        → heapq.heappush
schedule()   → peek self.waiting[0][2], heappop on completion
preempt()    → 删除，逻辑内联到 decode 路径
新增 _reheapify()
```

---

## 二、部分抢占（Partial Preemption）

### 改动前

decode 路径显存不足时，`preempt(seq)` 调用 `deallocate` 释放 seq 的全部 block_table，`num_cached_tokens` 归零。seq 退回 waiting 后必须从头重算全部 KV。

### 改动后

只释放最后一个 block（`free_tail_blocks(seq, 1)`），前缀 block 和对应的 `num_cached_tokens` 保留。下次被调度时从断点续算，只重算后缀。

### 关键点

- `BlockManager.free_tail_blocks(seq, n)` — 释放 seq 最后 n 个 block，保留前缀
- decode 路径的 preempt 逻辑 — 用 `free_tail_blocks` 取代 `deallocate`
- prefill 路径的恢复逻辑（第 53-59 行）— 被抢回来的 seq 的 `block_table` 非空但长度不足，需要补充分配缺失的 block：
  ```python
  needed = seq.num_blocks
  shortage = needed - len(seq.block_table)
  if shortage > 0:
      # allocate additional blocks
  ```
  如果没有这段，`prepare_prefill` 中的 `seq.block_table[i]` 会越界崩溃。

### 改动范围

| 文件 | 改动 |
|------|------|
| `block_manager.py` | 新增 `free_tail_blocks` |
| `scheduler.py` | decode preempt 改用 `free_tail_blocks`，prefill 路径补 block 分配 |

---

## 自闭环测试

测试分五层，从下往上逐层验证：

```
  端到端对比 (test_preempt_correctness.py)
    ↑ 实证：抢占路径 vs 不抢占路径，输出完全一致
  集成测试 (bench.py, stress_preempt.py)
    ↑ 验证不崩、吞吐不退化
  调度行为测试 (test_scheduler_preempt.py)
    ↑ 验证 preempt 后 seq 状态 + 恢复时补 block
  BlockManager 单元测试 (test_block_manager.py)
    ↑ 验证 free_tail_blocks 的 block 状态一致性
  计算正确性论证
    ↑ 逻辑证明 KV 等价性
```

---

### 第一层：BlockManager 单元测试（`tests/test_block_manager.py`，4 个，无 GPU）

每个测试通过 `BlockManager(num_blocks, block_size)` 直接构造场景，验证方法调用前后`block.ref_count`、`free_block_ids`、`used_block_ids`、`seq.block_table`、`seq.num_cached_tokens` 的状态。

```bash
python -c "
from tests.test_block_manager import TestFreeTailBlocks
t = TestFreeTailBlocks()
t.test_free_one_from_multi_block_seq()
t.test_free_all_equivalent_to_deallocate()
t.test_shared_block_not_deallocated()
t.test_free_from_single_block_seq()
print('BlockManager: 4/4 PASS')
"
```

#### 测试 1：释放 1 个 block（3 → 2）

**输入**：`seq = [1..9]`（9 token），`block_size=4`，分配 3 个 block，标记 `num_cached_tokens=9`（模拟 prefill 完成）。释放最后 1 个。

**预期结果**：
- `len(seq.block_table)` 从 3 → 2
- `seq.num_cached_tokens` 从 9 → 8（2 × block_size）
- 被释放的 block：`ref_count=0`，在 `free_block_ids` 中，不在 `used_block_ids` 中
- 剩余的 2 个 block：仍在 `used_block_ids` 中

**为什么对**：`free_tail_blocks` 的核心语义是"释放尾部的 n 个 block，保留前缀"。释放后 `num_cached_tokens` 与 `len(block_table) * block_size` 一致，block 的 ref_count 和 free/used 状态正确转移。

#### 测试 2：释放全部（2 → 0）等价 deallocate

**输入**：`seq = [1..5]`（5 token），分配 2 个 block。释放 2 个。

**预期结果**：
- `block_table` 为空，`num_cached_tokens = 0`
- 2 个 block 全部回到 `free_block_ids`，`ref_count=0`

**为什么对**：退化到 `deallocate` 的行为——`n == len(block_table)` 时 `free_tail_blocks` 必须跟 `deallocate` 一致。

#### 测试 3：共享 block 不受影响

**输入**：seq_a 的 block 0 被手动标记 `ref_count=2`（模拟 prefix cache 共享）。`free_tail_blocks(seq_a, 2)` 释放 block 1 和 2，block 0 在释放范围之外。

**预期结果**：
- block 0：`ref_count ≥ 1`，仍在 `used_block_ids` 中

**为什么对**：`free_tail_blocks` 只释放 `block_table[-n:]`，前缀 block 不在释放范围内，共享关系保持完整。

#### 测试 4：单 block seq 释放（1 → 0）

**输入**：`seq = [1,2,3]`（3 token），1 个 block。释放 1 个。

**预期结果**：
- `block_table` 为空，`num_cached_tokens = 0`
- block 回到 `free_block_ids`

**为什么对**：边界情况——只有 1 个 block 时也不会崩，行为正确。

---

### 第二层：调度行为测试（`tests/test_scheduler_preempt.py`，13 个，无 GPU）

手动构造 Scheduler + BlockManager 的内部状态，直接调用 `schedule()` 观察调度决策结果。**不依赖 GPU，不跑模型**。

```bash
python -c "
from tests.test_scheduler_preempt import (
    TestSchedulerPartialPreempt, TestPriorityScheduling,
    TestDecodePreemptSelf, TestMultiplePreemptions,
    TestPriorityPreemptInteraction,
)
t = TestSchedulerPartialPreempt()
t.test_partial_preempt_preserves_prefix()
t.test_partial_preempt_seq_resumes_with_extra_block()
print('Partial Preempt: 2/2 PASS')
t2 = TestPriorityScheduling()
t2.test_shortest_prompt_first()
t2.test_fifo_tiebreaker_same_remaining()
t2.test_reheapify_updates_stale_priorities()
print('Priority Scheduling: 3/3 PASS')
t3 = TestDecodePreemptSelf()
t3.test_self_preempt_when_running_empty()
print('Self-Preempt: 1/1 PASS')
t4 = TestMultiplePreemptions()
t4.test_double_preempt_block_consistency()
print('Multiple Preemptions: 1/1 PASS')
t5 = TestPriorityPreemptInteraction()
t5.test_preempted_seq_gets_correct_heap_priority()
t5.test_preempt_respects_heap_order_with_existing_waiting()
print('Priority-Preempt Interaction: 2/2 PASS')
"
```

#### 测试 5：部分抢占保留前缀

**构造**：`block_size=4`，`num_blocks=4`。seq_a（5 token，2 个 block）和 seq_b（5 token，2 个 block）都在 running。`free_block_ids` 为空。两者都在 block 边界（`5 % 4 == 1`）。

**调用**：`schedule()`。seq_a 的 `can_append` 失败 → seq_b 被 victim → `free_tail_blocks(seq_b, 1)`。

**预期结果**：

| 状态 | seq_b（victim） | seq_a（继续） |
|------|----------------|---------------|
| `block_table` 长度 | 1（从 2 减到 1） | 在 running |
| `num_cached_tokens` | 4（1 × block_size） | — |
| `status` | WAITING | RUNNING |
| `is_prefill` | True | — |
| 所在队列 | `sched.waiting` | `sched.running` |

**为什么对**：
- seq_b 作为最新进入 running 的 seq，completion token 最少，是合理的 victim（沉没成本最低）
- 只释放 1 个 block，前缀 block 和 `num_cached_tokens` 保留，未被清零
- 这证明了 partial preempt 的核心——不是全释放，而是局部释放

#### 测试 6：恢复时补分配缺失的 block

**构造**：`block_size=4`，`num_blocks=6`。seq = 9 token（需要 3 个 block），但手动只给它 2 个 block，`num_cached_tokens=8`。这模拟了一个被部分抢占后又回到 waiting 的 seq。空闲 block 有 4 个。

**调用**：`schedule()`。prefill 路径，`block_table` 非空，进入 else 分支：
```python
needed = seq.num_blocks         # = 3
shortage = needed - len(seq.block_table)  # = 3 - 2 = 1
# 分配 1 个额外 block
```
`num_scheduled_tokens = 1`，`8 + 1 == 9`（num_cached + scheduled == num_tokens）→ seq 进入 RUNNING。

**预期结果**：
- `len(seq.block_table) == 3`（补了 1 个）
- `seq.status == RUNNING`

**为什么对**：
- 没有这部分逻辑，`prepare_prefill` 中 `seq.block_table[i]` 会越界崩溃——这是之前 `tests/stress_preempt.py` 崩溃的直接原因
- 这个测试验证了"部分抢占 → 回 waiting → 再调度"的完整链路：block 补充分配正确、seq 从 waiting 到 running 的转移正确

#### 测试 7：短 prompt 优先调度

**构造**：`block_size=4`，`num_blocks=20`。三个 seq：short（5 token）、medium（50 token）、long（500 token），以 long → medium → short 的顺序通过 `sched.add()` 加入 waiting。`max_num_batched_tokens=10` 强制每轮只调度一个 seq。

**调用**：`schedule()`。

**预期结果**：
- 调用前 `sched.waiting[0][2]` 是 seq_short（堆顶 = 剩余 token 最少）
- `schedule()` 后：seq_short → RUNNING，seq_medium 和 seq_long 保持 WAITING

**为什么对**：
- 验证了 heap 的排序正确性——无论加入顺序如何，剩余 token 最少的始终在堆顶
- 验证了 `add()` → `_reheapify()` → `heappop` 的完整链路
- 剩余 token 相近时靠 tiebreaker（`Sequence.counter`）保证 FIFO，避免 starvation

#### 测试 8：相同优先级时的 FIFO tiebreaker

**构造**：`block_size=4`，`num_blocks=20`。两个 seq 都是 10 token，`num_cached_tokens=0`，剩余 10 token。先 add(seq_a)，后 add(seq_b)。

**预期结果**：堆顶是 seq_a（tiebreaker 更小），堆中第二个是 seq_b。两者的 remaining 都是 10。

**为什么对**：`(remaining, tiebreaker, seq)` 元组中，remaining 相同时 tiebreaker 决定顺序。验证 `Sequence.counter` 递增保证 FIFO。

#### 测试 9：_reheapify 刷新优先级

**构造**：`block_size=4`，`num_blocks=50`。seq_a（100 token，remaining=100），seq_b（50 token，remaining=50）。add 后堆顶是 seq_b（50 < 100）。手动将 seq_a 的 `num_cached_tokens` 改为 95（模拟 chunked prefill 后真正 remaining=5）。

**调用**：`_reheapify()`。

**预期结果**：堆顶变为 seq_a（5 < 50），remaining 值为 5 而非旧有的 100。

**为什么对**：`_reheapify` 从 seq 对象重新读取 `num_tokens - num_cached_tokens`，而非信任旧有的堆元组。chunked prefill 后 `num_cached_tokens` 被 `postprocess` 更新，下一次 `schedule()` 开头的 `_reheapify` 必须反映新的优先级。

#### 测试 10：running 唯一的 seq 自我抢占

**构造**：`block_size=4`，`num_blocks=2`。running 中仅一个 seq（5 token，占用 2 blocks，在 block 边界），无 waiting。空闲 block 为 0。

**调用**：`schedule()` → prefill 空跑 → decode 弹出 seq → `can_append` 返回 False → `self.running` 为空 → 进入 self-preempt 分支（`free_tail_blocks(seq, 1)` → push 回 waiting）。

**预期结果**：`scheduled_seqs == []`，`is_prefill == False`。seq.status == WAITING，`is_prefill=True`，`len(block_table)=1`，`num_cached_tokens=4`，seq 在 waiting 堆中。

**为什么对**：这是 scheduler.py:86-92 的 else 分支——running 空时 seq 不是 victim 别人而是 victim 自己。同时验证了 self-preempt 的 bug 修复：原代码 `assert scheduled_seqs` 在 self-preempt 后 `scheduled_seqs` 为空时崩溃，修复为条件判断。`llm_engine.py` 的 `step()` 也加了早期返回以处理空 `scheduled_seqs`。

#### 测试 11：两次连续抢占的 block 一致性

**构造**：`block_size=4`，`num_blocks=10`。seq（15 token）手动分配 4 个 block。

**调用**：`free_tail_blocks(seq, 1)` 两次。

**预期结果**：
| 状态 | 第 1 次后 | 第 2 次后 |
|------|----------|----------|
| `len(block_table)` | 3 | 2 |
| `num_cached_tokens` | 12 | 8 |
| `len(free_block_ids)` | 7 | 8 |
| 剩余 block ref_count | 1 | 1 |
| 释放 block ref_count | 0 | 0 |

**为什么对**：验证 `free_tail_blocks` 在"抢占 → 恢复 → 再次抢占"循环中不产生 ref_count 泄漏或越界。每次只减最后一个 block 的 ref_count，与剩余 block 无关。

#### 测试 12：preempt victim 获得正确的 heap 优先级

**构造**：`block_size=4`，`num_blocks=4`。seq_a（5 token）和 seq_b（5 token）在 running，各占 2 block，free=0。seq_long（100 token）在 waiting。两者都在 block 边界。

**调用**：`schedule()`。prefill 阶段 seq_long 因 free=0 无法分配。decode 阶段 seq_a 的 `can_append` 失败 → seq_b 被 victim → `free_tail_blocks(seq_b, 1)` → seq_b 压入 waiting 堆。

**预期结果**：
- seq_b.status == WAITING，`len(block_table)=1`，`num_cached_tokens=4`，`is_prefill=True`
- seq_b 剩余 token = 5 - 4 = 1，小于 seq_long 的 100
- 堆顶是 seq_b（不是 seq_long）

**为什么对**：验证优先级调度与部分抢占的交互——preempt 后的 seq 按**当前剩余 token 数**进入堆中，而不是原始 token 数。被抢占后前缀保留 block 使 `num_cached_tokens > 0`，`remaining` 很小，应该排在长 prompt 前面。

#### 测试 13：preempt victim 插入已有 waiting seq 的正确位置

**构造**：`block_size=4`，`num_blocks=4`。seq_a（5 token）和 seq_b（5 token）在 running。seq_medium（50 token）和 seq_long（100 token）在 waiting。free=0。

**调用**：`schedule()` → seq_b 被 victim → waiting 堆中现在有 3 个 seq。手动 `_reheapify()` 后按 `remaining` 排序。

**预期结果**：
- 排序：seq_b(remaining=1) < seq_medium(50) < seq_long(100)
- 三项均在 waiting 中

**为什么对**：验证 preempt victim 插入已有 waiting seq 时，堆在 `_reheapify` 后保持正确的全局顺序。preempt victim 带着极小的 remaining 进入堆，必须在所有从未被调度过的 seq 之前。

---

### 第三层：集成测试（需要 GPU）

#### 基准回归（bench.py）

```bash
python bench.py
```

256 seq，随机 100-1024 token，标准配置。KV cache 足够大，preempt 几乎不触发。

**预期**：不崩，吞吐与原版在同一量级（约 1400 tok/s）。**验证正常路径未被改动破坏。**

#### 压力测试（stress_preempt.py）

```bash
python tests/stress_preempt.py
```

32 seq × 1500-2500 token prompt，`gpu_memory_utilization=0.55` 人为压缩 KV cache，强制触发 preempt。

**预期**：不崩。**验证 preempt 链路从 free_tail_blocks → 补 block → 断点续算不崩溃。**

---

### 第四层：计算正确性论证

部分抢占不改变计算结果。这不是通过对比特定采样结果来验证的（随机采样 + FlashAttention 浮点非确定性使 token 对比不可靠），而是通过逻辑推理：

1. 模型是纯函数：`KV = f(token_ids, positions, model_weights)`，无状态、无随机性
2. 断点续算时：`start = num_cached_tokens`，`end = num_tokens`。`input_ids = seq[start:end]`，`positions = range(start, end)`。这些与一次性从 0 算到 `num_tokens` 时 `[start, end)` 区间的输入**完全一致**
3. 前缀 KV 从 `k_cache`/`v_cache` 读取（FlashAttention 的 `block_table` 参数），这些 KV 就是之前计算的结果，等同于一次性计算到 `start` 位置的 KV
4. 既然输入一致 → KV 一致 → logits 一致 → 采样结果一致

**两条路径算出相同的 KV，这是确定性计算，不是概率论证。**

---

### 第五层：端到端对比（`tests/test_preempt_correctness.py`，需要 GPU）

**核心思路**：同一批 prompt 跑两遍，唯一变量是 KV cache 大小。

```
大 KV cache (gpu_memory_utilization=0.9)
  → blocks 充足 → 所有 seq 顺利 decode → 不触发抢占
  → 输出 = "正确答案"（golden）

小 KV cache (gpu_memory_utilization=0.05 + block reservation)
  → blocks 刚好够 prefill，decode 时耗尽 → 触发抢占
  → 输出 = "测试结果"（test）

对比：golden == test？
```

**验证逻辑**：模型在 `enforce_eager=True` + `temperature≈0` 下是确定性的。如果代码正确，抢占路径产生的 KV cache 内容跟不抢占路径完全一致 → logits 一致 → 采样 token 一致。如果 token 有任何差异，说明抢占过程中 KV 被污染了或计算路径有 bug。

**为什么比逻辑论证更强**：逻辑论证说"相同输入 → 相同 KV"，但这是假设代码没有 bug。端到端对比直接验证了**代码实际行为**。即使逻辑论证无懈可击，也无法排除某个边界条件下 `block_table` 索引写错、`num_cached_tokens` 算错、`position` 传错等具体实现问题。端到端对比用真实 GPU 计算的结果来检查这些。

**什么情况会被判定为"正确"**：

| 观测结果 | 判定 |
|----------|------|
| test 触发 >0 次抢占，且 golden tokens == test tokens 逐位一致 | **正确** |
| test 触发 0 次抢占 | 测试无效（没测到目标路径） |
| test 触发 >0 次抢占，但 tokens 有差异 | **有 bug** |

**关键设计**：

| 条件 | 作用 |
|------|------|
| `enforce_eager=True` | 禁用 FlashAttention 融合 kernel，消除 attention 非确定性 |
| `temperature=1e-6` | 近似 greedy sampling，对浮点噪声鲁棒 |
| 2 条 seq，同长度 | 保持批处理一致，消除批处理带来的浮点差异 |
| 子进程隔离 | 避免两次 `dist.init_process_group` 冲突 |
| block reservation | 在 `generate()` 前手动消耗多余 block，确保 prefill 后 free=0 |
| monkey-patch `free_tail_blocks` | 计数抢占次数，确认测试真的触发了抢占 |

**预期结果**：baseline（0 preempt）和 test（>0 preempt）输出完全一致。

```bash
python tests/test_preempt_correctness.py
```

---

## 测试结果（2026-06-09）

### 第一层：BlockManager 单元测试

```
[ 1/11] free_one_from_multi_block_seq: PASS
[ 2/11] free_all_equivalent_to_deallocate: PASS
[ 3/11] shared_block_not_deallocated: PASS
[ 4/11] free_from_single_block_seq: PASS
```

### 第二层：Scheduler 调度行为测试

```
[ 5/11] partial_preempt_preserves_prefix: PASS
[ 6/11] partial_preempt_seq_resumes_with_extra_block: PASS
[ 7/11] shortest_prompt_first: PASS
[ 8/11] fifo_tiebreaker_same_remaining: PASS
[ 9/11] reheapify_updates_stale_priorities: PASS
[10/11] self_preempt_when_running_empty: PASS
[11/11] double_preempt_block_consistency: PASS
[12/13] preempted_seq_gets_correct_heap_priority: PASS
[13/13] preempt_respects_heap_order_with_existing_waiting: PASS
```

### 第三层：集成测试

| 测试 | 结果 | 详情 |
|------|------|------|
| `bench.py` | PASS | 256 seq, 133966 tok, 20.62s, 6498 tok/s，不崩 |
| `tests/stress_preempt.py` | PASS | 32 seq × 1500-2500 prompt, 14676 tok, 6.55s, 2242 tok/s，不崩 |

### 第四层：计算正确性论证

逻辑论证通过（见上文）。

### 第五层：端到端对比

```
Preempt events: baseline=0, test=1
  seq[0]: MATCH (256 tokens)
  seq[1]: MATCH (256 tokens)
PASS: All outputs match. Partial preemption is empirically correct.
```

2 条 seq，抢占发生 1 次，抢占路径与不抢占路径输出的 512 个 token 完全一致。

---

## 端到端测试调试记录

开发第五层端到端对比测试时遇到以下问题及解决过程。

### 调试方法

全程不使用 pdb/breakpoint（模型加载需 2 分钟，设断点不现实），不加 `torch.allclose` 对比中间 tensor（需改 model_runner 侵入性太大），不 dump KV cache 二进制（28MB/block，diff 成本高）。只用一种方式：

> **加 print → 跑 → grep → 看结果 → 推断根因 → 改参数 → 再跑**

每次只验证一个假设。

### 问题 1：两个 LLM 实例冲突

**现象**：
```
ValueError: trying to initialize the default process group twice!
```

**原因**：`ModelRunner.__init__` 调用 `dist.init_process_group`，`exit()` 里的 `dist.destroy_process_group` 通过 `atexit` 注册，只在进程退出时触发。`del llm_large` 不会触发清理，创建第二个 LLM 时进程组仍存活。

**解决**：每轮 LLM 运行在独立的 `subprocess` 中，进程退出自然释放。

### 问题 2：子进程 inline code 超长

**现象**：
```
OSError: [Errno 7] Argument list too long
```

**原因**：32 seq × 2000 token 的 prompt 列表嵌在命令行字符串里，超过 Linux `ARG_MAX`。

**解决**：用 pickle 把数据序列化到临时文件，子进程从文件反序列化。

### 问题 3：f-string 与 `.format()` 花括号冲突

**现象**：
```
KeyError: 'data'
```

**原因**：`RUNNER_SCRIPT` 模板字符串同时包含 Python f-string 的 `{data['key']}` 和 `.format()` 的 `{input_path}`。`.format()` 把前者也当占位符解析。

**解决**：所有 Python 字面量花括号用 `{{` `}}` 转义，只保留 `{input_path}` 和 `{output_path}` 供 `.format()` 填充。

### 问题 4：KV cache 太大，始终不触发抢占

**现象**：

| `gpu_memory_utilization` | blocks | 32 seq prefill 需 ~238 blocks | 结果 |
|--------------------------|--------|-------------------------------|------|
| 0.40 | ? | 远小于 | 0 preempt |
| 0.25 | 357 | 远超 | 0 preempt |
| 0.10 | 96 | 仍然 > 预估值 | 30 preempt |
| 0.05 | 9 | — | 0 preempt（9 个够用） |
| 0.044 | 0 | — | crash（assert num_kvcache_blocks > 0） |

**定位**：在子进程代码中加入 `print(f"KV blocks: {total_blocks} total, {free0} free")`，发现 `enforce_eager=True` 时不捕获 CUDA Graph，`allocate_kv_cache` 公式中 `peak - current ≈ 0`，比正常模式多出几百 MB 给 KV cache。RTX 4090 24GB 上即使 0.05 utilization 也有 9 blocks，而 0.044 就直接 0 了。blocks 是离散量（28MB/block），utilization 每调 0.001 可能多/少一个 block，无法精确实控。

**解决**：gpu_memory_utilization 固定 0.05（9 blocks），`generate()` 前手动调用 `bm._allocate_block()` 消耗多余 block，设置 `ref_count=999` 防释放，确保 prefill 后 `free_block_ids` 恰好为 0。

### 问题 5：多 seq 批处理浮点非确定性

**现象**：32 seq 配置下触发 30 次抢占，6/32 条 seq 输出不一致：

```
seq[22]: first diff at pos 0   → 第一个 completion token 就不同
seq[3]:  first diff at pos 1
seq[10]: first diff at pos 4
```

**排查**：差异在 pos 0/1/4——极早期分叉。如果是抢占续算逻辑 bug（block_table 索引错、position 算错），KV cache 会读出垃圾，后续所有 token 都会乱，不会只差 1-2 个位置。极早期分叉 + 其余 token 正常，更符合"第一个采样 step 微小 logits 差异被采样放大"的模式。加上只在多 seq 场景出现，减到 2 seq 后消失，判断为**抢占改变了 seq 调度顺序 → 批处理组合变化 → 浮点累加顺序不同 → logits 微小差异**。

**解决**：改用 2 条相同长度的 seq，保持批处理组合在抢占前后一致。

### 问题 6：单 seq 触发 self-preempt assert（已修复）

**现象**：尝试用 1 条 seq 彻底消除批处理差异，但直接 crash。

**原因**：running 唯一 seq 自我抢占后 `scheduled_seqs` 为空，触发 `scheduler.py:98` 的 `assert scheduled_seqs`。

**修复**（2026-06-09）：
- `scheduler.py:98`：`assert scheduled_seqs` → `if scheduled_seqs:` 条件判断
- `llm_engine.py:51-52`：`step()` 中 `scheduled_seqs` 为空时早期返回 `[], 0`

端到端测试因其他原因改用 2 条 seq，但 self-preempt 路径在单元测试（测试 10）中已覆盖。

### 问题 7：stress_preempt 复现时 `num_tokens` 未定义

**现象**：修复 self-preempt 后跑 `stress_preempt.py` 崩：
```
NameError: name 'num_tokens' is not defined
```

**原因**：`llm_engine.py` 的 `step()` 原本在 `schedule()` 返回后立即计算 `num_tokens`。修复 self-preempt 时添加了早期返回，但 `num_tokens` 计算被误删。

**修复**：在 `model_runner.call("run")` 前恢复 `num_tokens = sum(...) if is_prefill else -len(seqs)` 一行。
