from collections import deque
import heapq

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: list[Sequence] = []   # heap: (remaining, tiebreaker, seq)
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        remaining = seq.num_tokens - seq.num_cached_tokens
        heapq.heappush(self.waiting, (remaining, next(Sequence.counter), seq))

    def _reheapify(self):
        """Rebuild heap with current seq priorities."""
        entries = [(s.num_tokens - s.num_cached_tokens, next(Sequence.counter), s)
                   for _, _, s in self.waiting]
        self.waiting = entries
        heapq.heapify(self.waiting) if self.waiting else None

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        self._reheapify()

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0][2]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                needed = seq.num_blocks
                shortage = needed - len(seq.block_table)
                if shortage > 0:
                    if len(self.block_manager.free_block_ids) < shortage:
                        break
                    for _ in range(shortage):
                        seq.block_table.append(self.block_manager._allocate_block())
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                heapq.heappop(self.waiting)
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.block_manager.free_tail_blocks(victim, 1)
                    victim.status = SequenceStatus.WAITING
                    victim.is_prefill = True
                    remaining = victim.num_tokens - victim.num_cached_tokens
                    heapq.heappush(self.waiting, (remaining, next(Sequence.counter), victim))
                else:
                    self.block_manager.free_tail_blocks(seq, 1)
                    seq.status = SequenceStatus.WAITING
                    seq.is_prefill = True
                    remaining = seq.num_tokens - seq.num_cached_tokens
                    heapq.heappush(self.waiting, (remaining, next(Sequence.counter), seq))
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
