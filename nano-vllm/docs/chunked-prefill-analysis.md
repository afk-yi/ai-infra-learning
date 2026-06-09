# KV Cache 部分抢占优化

## 问题

显存不足时，当前 preempt 策略是**整条 seq 的 block_table 全部释放**，`num_cached_tokens` 归零。seq 退回 waiting 后必须从头重算全部 KV，沉没成本完全丢弃。

## 当前实现

两处触发 preempt：

**Prefill 路径**（`scheduler.py:37-38`）：

```python
if num_cached_blocks == -1:   # can_allocate 失败，显存不够
    break                      # 不抢占，直接放弃本轮 prefill
```

**Decode 路径**（`scheduler.py:59-65`）：

```python
seq = self.running.popleft()
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())   # 抢最新的 running seq
    else:
        self.preempt(seq)                  # 没人可抢，牺牲自己
        break
```

`preempt` 方法（`scheduler.py:75-79`）：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)    # 全部 block 释放
    self.waiting.appendleft(seq)          # num_cached_tokens 归零
```

## Victim 选择分析

`running.pop()` 取的是最新加入的 seq——刚做完 prefill，completion token 最少（≈0）。选它做 victim 是合理的：沉没成本最低，且占用的 block 也最少。

所以 victim 选择不需要改。

## 改进：部分释放代替全部释放

只释放 victim 的最后一个 block，保留前缀 block 和对应的 `num_cached_tokens`。victim 退回 waiting 后，下次被调度时从断点继续，只重算最后几百个 token。

### BlockManager 改动

新增 `free_tail_blocks` 方法（`block_manager.py`）：

```python
def free_tail_blocks(self, seq: Sequence, n: int):
    """释放 seq 最后 n 个 block"""
    for block_id in seq.block_table[-n:]:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.block_table = seq.block_table[:-n]
    seq.num_cached_tokens = len(seq.block_table) * self.block_size
```

逻辑与现有 `deallocate` 完全一致，只遍历最后 n 个 block。

### Scheduler 改动

decode 路径的 preempt 逻辑（`scheduler.py:59-65`）改为：

```python
seq = self.running.popleft()
while not self.block_manager.can_append(seq):
    if self.running:
        victim = self.running.pop()
        self.block_manager.free_tail_blocks(victim, 1)   # 只抢最后 1 个 block
        victim.status = SequenceStatus.WAITING
        victim.is_prefill = True
        self.waiting.appendleft(victim)
    else:
        self.block_manager.free_tail_blocks(seq, 1)       # 只放自己最后 1 个 block
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.appendleft(seq)
        break
```

原来的 `preempt` 方法可以删除。

### 边界情况

- victim 只有 1 个 block：`free_tail_blocks(seq, 1)` 等价于全部释放，`block_table` 变空、`num_cached_tokens` 归零，行为跟原来一致
- victim 的前缀不会共享给其他 seq：保留不是为了共享，是为了减少 victim 自身的重算量

## 效果对比

```
场景：256 seq running，显存满了，需要腾 1 个 block 给 seq X 的 decode

当前：选最新 seq Z（prompt=800, 已生成 3 token, 共 4 个 block）
     Z 全部 4 个 block 释放，num_cached_tokens 归零
     重调度时重算全部 803 token

改进：选最新 seq Z
     只释放最后 1 个 block，保留前 3 个 block（768 token）
     num_cached_tokens = 768
     重调度时只重算 803 - 768 = 35 token
```

## 影响范围

| 文件 | 改动 |
|------|------|
| `block_manager.py` | 新增 `free_tail_blocks` 方法 |
| `scheduler.py` | decode preempt 路径改用 `free_tail_blocks`，删除原 `preempt` |

不涉及 sequence.py、model_runner.py、config.py。
