from collections import deque
import heapq

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler

BS = 4  # block_size for tests


def make_seq(token_ids):
    return Sequence(token_ids)


class TestSchedulerPartialPreempt:

    def setup_method(self):
        Sequence.block_size = BS

    def test_partial_preempt_preserves_prefix(self):
        """When decode runs out of blocks, victim keeps prefix KV cache."""
        self.setup_method()
        num_blocks = 4

        bm = BlockManager(num_blocks, BS)

        # seq_a: 5 tokens → 2 blocks, at block boundary (5 % 4 == 1)
        seq_a = make_seq(list(range(5)))
        bm.allocate(seq_a, num_cached_blocks=0)
        seq_a.num_cached_tokens = 5
        seq_a.status = SequenceStatus.RUNNING
        seq_a.is_prefill = False

        # seq_b: 5 tokens → 2 blocks, at block boundary
        seq_b = make_seq(list(range(5, 10)))
        bm.allocate(seq_b, num_cached_blocks=0)
        seq_b.num_cached_tokens = 5
        seq_b.status = SequenceStatus.RUNNING
        seq_b.is_prefill = False

        assert len(bm.free_block_ids) == 0  # 2 + 2 = 4 blocks consumed

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []                         # heap, not deque
        sched.running = deque([seq_a, seq_b])

        # seq_a at block boundary → can_append fails → preempt seq_b
        sched.schedule()

        # seq_b: partially preempted, keeps prefix blocks
        waiting_seqs = [e[2] for e in sched.waiting]
        assert seq_b in waiting_seqs
        assert len(seq_b.block_table) == 1     # had 2, lost 1 tail
        assert seq_b.num_cached_tokens == BS   # 1 * BS
        assert seq_b.status == SequenceStatus.WAITING
        assert seq_b.is_prefill is True

        # seq_a: got the freed block and was scheduled for decode
        assert seq_a in sched.running

    def test_partial_preempt_seq_resumes_with_extra_block(self):
        """A seq with shortened block_table gets missing blocks on reschedule."""
        self.setup_method()
        num_blocks = 6

        bm = BlockManager(num_blocks, BS)

        # seq: 9 tokens → needs 3 blocks. Manually give it only 2 (simulating
        # a partial preempt that kept prefix but freed tail).
        seq = make_seq(list(range(9)))
        seq.block_table = [bm._allocate_block(), bm._allocate_block()]
        seq.num_cached_tokens = 8  # 2 blocks * BS
        seq.status = SequenceStatus.WAITING

        assert len(bm.free_block_ids) == 4  # 6 - 2

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        # Push into heap properly: (remaining, tiebreaker, seq)
        sched.waiting = []
        heapq.heappush(sched.waiting,
                       (seq.num_tokens - seq.num_cached_tokens,
                        next(Sequence.counter), seq))
        sched.running = deque()

        sched.schedule()

        # seq got 1 more block and completed prefill
        assert len(seq.block_table) == 3       # was 2, got 1
        assert seq.num_cached_tokens == 8      # unchanged until postprocess
        assert seq.status == SequenceStatus.RUNNING


class TestPriorityScheduling:

    def setup_method(self):
        Sequence.block_size = BS

    def test_shortest_prompt_first(self):
        """Schedule picks shortest prompt from waiting queue first."""
        self.setup_method()
        num_blocks = 20

        bm = BlockManager(num_blocks, BS)

        seq_short = make_seq(list(range(5)))     # 5 tokens
        seq_medium = make_seq(list(range(50)))    # 50 tokens
        seq_long = make_seq(list(range(500)))     # 500 tokens

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 10  # tight: only shortest fits
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque()

        # Add in reverse order: longest first
        sched.add(seq_long)
        sched.add(seq_medium)
        sched.add(seq_short)

        # Peek: heap top should be shortest (smallest remaining)
        _, _, top = sched.waiting[0]
        assert top is seq_short, f"expected short (5 tok), got {top.num_tokens} tok"

        # Schedule: shortest should complete prefill first
        sched.schedule()

        assert seq_short.status == SequenceStatus.RUNNING
        assert seq_medium.status == SequenceStatus.WAITING
        assert seq_long.status == SequenceStatus.WAITING

    def test_fifo_tiebreaker_same_remaining(self):
        """When two seqs have same remaining tokens, earlier add wins (FIFO)."""
        self.setup_method()
        num_blocks = 20

        bm = BlockManager(num_blocks, BS)
        seq_a = make_seq(list(range(10)))  # 10 tokens
        seq_b = make_seq(list(range(10, 20)))  # also 10 tokens

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque()

        sched.add(seq_a)
        sched.add(seq_b)

        # Both have remaining=10, seq_a added first → should be on top
        rem_a, tie_a, top = sched.waiting[0]
        rem_b, tie_b, _ = sched.waiting[1]
        assert top is seq_a, f"FIFO broken: expected seq_a on top, got {top.num_tokens}"
        assert tie_a < tie_b, f"tiebreaker: a={tie_a} should be < b={tie_b}"
        assert rem_a == rem_b == 10

    def test_reheapify_updates_stale_priorities(self):
        """After chunked prefill changes num_cached_tokens, _reheapify fixes heap order."""
        self.setup_method()
        num_blocks = 50

        bm = BlockManager(num_blocks, BS)
        seq_a = make_seq(list(range(100)))  # 100 tokens → remaining 100
        seq_b = make_seq(list(range(50)))   # 50 tokens → remaining 50

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque()

        sched.add(seq_a)
        sched.add(seq_b)
        # Heap: [(50, tb, seq_b), (100, ta, seq_a)] — B on top
        _, _, top_before = sched.waiting[0]
        assert top_before is seq_b, f"expected B (50) on top, got {top_before.num_tokens}"

        # Simulate chunked prefill: A got 95 tokens cached, now only 5 remain
        seq_a.num_cached_tokens = 95
        # seq_b unchanged: still 50 remaining

        sched._reheapify()
        # After reheapify: A has remaining=5 (< 50), should be on top
        remaining, _, top_after = sched.waiting[0]
        assert top_after is seq_a, f"expected A (5) on top after reheapify, got {top_after.num_tokens}"
        assert remaining == 5


class TestDecodePreemptSelf:

    def setup_method(self):
        Sequence.block_size = BS

    def test_self_preempt_when_running_empty(self):
        """When the only running seq cannot append, it preempts itself."""
        self.setup_method()
        num_blocks = 2

        bm = BlockManager(num_blocks, BS)
        # seq: 5 tokens → 2 blocks at block_size=4. Exhaust all blocks.
        seq = make_seq(list(range(5)))
        bm.allocate(seq, num_cached_blocks=0)
        seq.num_cached_tokens = 5
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False

        assert len(bm.free_block_ids) == 0  # all used

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque([seq])

        # schedule → decode: seq popleft, can_append fails, running empty → self-preempt
        # scheduled_seqs ends up empty, schedule returns ([], False)
        scheduled_seqs, is_prefill = sched.schedule()
        assert scheduled_seqs == []
        assert is_prefill is False

        # Seq self-preempted: lost 1 tail block, now in waiting
        assert seq.status == SequenceStatus.WAITING
        assert seq.is_prefill is True
        assert len(seq.block_table) == 1       # was 2, lost 1
        assert seq.num_cached_tokens == BS     # 1 * BS
        assert len(bm.free_block_ids) == 1     # 1 block freed
        waiting_seqs = [e[2] for e in sched.waiting]
        assert seq in waiting_seqs


class TestMultiplePreemptions:

    def setup_method(self):
        Sequence.block_size = BS

    def test_double_preempt_block_consistency(self):
        """Preempting a seq twice keeps block ref_counts consistent."""
        self.setup_method()
        num_blocks = 10

        bm = BlockManager(num_blocks, BS)
        # seq: 15 tokens → needs 4 blocks (ceil(15/4))
        seq = make_seq(list(range(15)))
        seq.block_table = [bm._allocate_block() for _ in range(4)]
        seq.num_cached_tokens = 15

        assert len(bm.free_block_ids) == 6

        # First preempt: release 1 tail
        bm.free_tail_blocks(seq, 1)
        assert len(seq.block_table) == 3
        assert seq.num_cached_tokens == 12  # 3 * BS
        assert len(bm.free_block_ids) == 7

        # Second preempt: release another tail
        bm.free_tail_blocks(seq, 1)
        assert len(seq.block_table) == 2
        assert seq.num_cached_tokens == 8   # 2 * BS
        assert len(bm.free_block_ids) == 8

        # Remaining blocks still have ref_count=1
        for bid in seq.block_table:
            assert bm.blocks[bid].ref_count == 1
        # Freed blocks have ref_count=0 and are in free list
        assert all(bm.blocks[bid].ref_count == 0 for bid in bm.free_block_ids)


class TestPriorityPreemptInteraction:

    def setup_method(self):
        Sequence.block_size = BS

    def test_preempted_seq_gets_correct_heap_priority(self):
        """Victim of partial preempt goes to waiting with updated (short) remaining."""
        self.setup_method()
        num_blocks = 4

        bm = BlockManager(num_blocks, BS)

        # Two seqs in running, each 5 tokens → 2 blocks. All 4 blocks used.
        seq_a = make_seq(list(range(5)))
        bm.allocate(seq_a, num_cached_blocks=0)
        seq_a.num_cached_tokens = 5
        seq_a.status = SequenceStatus.RUNNING
        seq_a.is_prefill = False

        seq_b = make_seq(list(range(5, 10)))
        bm.allocate(seq_b, num_cached_blocks=0)
        seq_b.num_cached_tokens = 5
        seq_b.status = SequenceStatus.RUNNING
        seq_b.is_prefill = False

        assert len(bm.free_block_ids) == 0

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque([seq_a, seq_b])

        # Long seq in waiting: 100 tokens, remaining=100
        seq_long = make_seq(list(range(1000, 1100)))
        sched.add(seq_long)

        # Schedule: prefill can't allocate for seq_long (needs 25 blocks, free=0).
        # Decode: seq_a at boundary (5%4=1), can_append fails, running has seq_b → victim.
        # seq_b: partial preempt, 2→1 block, num_cached_tokens=4, remaining=1.
        # seq_a gets the freed block.
        sched.schedule()

        # seq_b should be victim with partial preempt
        assert seq_b.status == SequenceStatus.WAITING
        assert len(seq_b.block_table) == 1
        assert seq_b.num_cached_tokens == BS  # 4
        assert seq_b.is_prefill is True

        # seq_b remaining = 5 - 4 = 1, which is < seq_long's 100
        # Heap: [(1, seq_b), (100, seq_long)] → seq_b on top
        _, _, top = sched.waiting[0]
        assert top is seq_b, f"expected seq_b (remaining=1) on top, got remaining={top.num_tokens - top.num_cached_tokens}"

    def test_preempt_respects_heap_order_with_existing_waiting(self):
        """Preempt victim inserts into waiting at correct heap position."""
        self.setup_method()
        num_blocks = 4

        bm = BlockManager(num_blocks, BS)

        seq_a = make_seq(list(range(5)))
        bm.allocate(seq_a, num_cached_blocks=0)
        seq_a.num_cached_tokens = 5
        seq_a.status = SequenceStatus.RUNNING
        seq_a.is_prefill = False

        seq_b = make_seq(list(range(5, 10)))
        bm.allocate(seq_b, num_cached_blocks=0)
        seq_b.num_cached_tokens = 5
        seq_b.status = SequenceStatus.RUNNING
        seq_b.is_prefill = False

        assert len(bm.free_block_ids) == 0

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 8192
        config.eos = -1
        config.kvcache_block_size = BS
        config.num_kvcache_blocks = num_blocks

        sched = Scheduler.__new__(Scheduler)
        sched.max_num_seqs = config.max_num_seqs
        sched.max_num_batched_tokens = config.max_num_batched_tokens
        sched.eos = config.eos
        sched.block_size = config.kvcache_block_size
        sched.block_manager = bm
        sched.waiting = []
        sched.running = deque([seq_a, seq_b])

        # Two waiting seqs: medium (50 tokens, remaining=50) and long (100 tokens, remaining=100)
        seq_medium = make_seq(list(range(100, 150)))
        sched.add(seq_medium)
        seq_long = make_seq(list(range(1000, 1100)))
        sched.add(seq_long)

        # Heap: [(50, seq_medium), (100, seq_long)]
        _, _, top = sched.waiting[0]
        assert top is seq_medium

        # Schedule: prefill can't allocate (not enough free blocks for either waiting seq).
        # Decode: seq_a at boundary, can_append fails → victim seq_b.
        # seq_b: partial preempt → remaining=1.
        sched.schedule()

        # seq_b now in waiting heap with remaining=1
        # After _reheapify in next schedule:
        # Heap order: (1, seq_b), (50, seq_medium), (100, seq_long)
        # But _reheapify is called at the START of schedule, so it happens
        # when we call schedule() again. Let's check manually:
        sched._reheapify()
        entries = sorted(sched.waiting)  # sorted by (remaining, tiebreaker, seq)
        assert entries[0][2] is seq_b, f"expected seq_b (rem=1) first, got {entries[0][2].num_tokens}"
        assert entries[1][2] is seq_medium, f"expected seq_medium (rem=50) second"
        assert entries[2][2] is seq_long, f"expected seq_long (rem=100) third"
