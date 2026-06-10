from collections import deque
import heapq

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler

BS = 4  # block_size for tests


def make_seq(token_ids):
    return Sequence(token_ids)


def make_config(num_blocks, max_batched=8192, max_seqs=512):
    config = Config.__new__(Config)
    config.max_num_seqs = max_seqs
    config.max_num_batched_tokens = max_batched
    config.eos = -1
    config.kvcache_block_size = BS
    config.num_kvcache_blocks = num_blocks
    return config


def make_scheduler(config, bm, waiting=None, running=None):
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = config.max_num_seqs
    sched.max_num_batched_tokens = config.max_num_batched_tokens
    sched.eos = config.eos
    sched.block_size = config.kvcache_block_size
    sched.block_manager = bm
    sched.waiting = waiting if waiting is not None else []
    sched.running = running if running is not None else deque()
    return sched


class TestMixedBatchBasic:

    def setup_method(self):
        Sequence.block_size = BS

    def test_mixed_batch_basic(self):
        """1 waiting + 1 running -> both scheduled in same step."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        # Waiting seq: 8 tokens -> 2 blocks, fresh prefill
        seq_waiting = make_seq(list(range(8)))
        assert seq_waiting.is_prefill is True

        # Running seq: already in decode, 5 tokens, 2 blocks
        seq_running = make_seq(list(range(100, 105)))
        bm.allocate(seq_running, num_cached_blocks=0)
        seq_running.num_cached_tokens = 5
        seq_running.status = SequenceStatus.RUNNING
        seq_running.is_prefill = False

        config = make_config(num_blocks)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_running]))

        # Add waiting seq properly via heap
        sched.add(seq_waiting)

        scheduled, has_prefill = sched.schedule()

        # Both seqs should be scheduled
        assert seq_waiting in scheduled, "waiting seq not scheduled"
        assert seq_running in scheduled, "running seq not scheduled"
        # has_prefill=True because waiting seq is prefill
        assert has_prefill is True
        # seq_waiting: prefill, seq_running: decode
        assert seq_waiting.is_prefill is True
        assert seq_running.is_prefill is False
        # seq_waiting scheduled_tokens > 1, seq_running scheduled_tokens == 1
        assert seq_waiting.num_scheduled_tokens == 8
        assert seq_running.num_scheduled_tokens == 1
        # seq_waiting completed prefill -> should be in running (via fresh_running)
        assert seq_waiting.status == SequenceStatus.RUNNING

    def test_token_budget_split(self):
        """Prefill + decode share the token budget correctly."""
        self.setup_method()
        num_blocks = 30
        bm = BlockManager(num_blocks, BS)

        # Large waiting seq: 20 tokens -> 5 blocks
        seq_waiting = make_seq(list(range(20)))
        seq_running = make_seq(list(range(100, 105)))
        bm.allocate(seq_running, num_cached_blocks=0)
        seq_running.num_cached_tokens = 5
        seq_running.status = SequenceStatus.RUNNING
        seq_running.is_prefill = False

        config = make_config(num_blocks, max_batched=12)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_running]))
        sched.add(seq_waiting)

        scheduled, has_prefill = sched.schedule()

        # Budget=12: prefill uses 12 (chunked), decode gets 0
        assert seq_waiting.num_scheduled_tokens == 12
        # seq_running not scheduled because budget exhausted by prefill
        assert seq_running.num_scheduled_tokens == 0
        assert seq_running not in scheduled
        assert has_prefill is True

    def test_pure_decode_fallback(self):
        """waiting empty -> same as current decode-only step."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        seq_a = make_seq(list(range(5)))
        bm.allocate(seq_a, num_cached_blocks=0)
        seq_a.num_cached_tokens = 5
        seq_a.status = SequenceStatus.RUNNING
        seq_a.is_prefill = False

        seq_b = make_seq(list(range(10, 15)))
        bm.allocate(seq_b, num_cached_blocks=0)
        seq_b.num_cached_tokens = 5
        seq_b.status = SequenceStatus.RUNNING
        seq_b.is_prefill = False

        config = make_config(num_blocks)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_a, seq_b]))

        scheduled, has_prefill = sched.schedule()

        assert len(scheduled) == 2
        assert has_prefill is False
        assert all(s.num_scheduled_tokens == 1 for s in scheduled)
        assert all(not s.is_prefill for s in scheduled)

    def test_pure_prefill_fallback(self):
        """running empty -> same as current prefill-only step."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        seq = make_seq(list(range(8)))

        config = make_config(num_blocks)
        sched = make_scheduler(config, bm)
        sched.add(seq)

        scheduled, has_prefill = sched.schedule()

        assert len(scheduled) == 1
        assert has_prefill is True
        assert seq.is_prefill is True
        assert seq.num_scheduled_tokens == 8
        # Freshly prefilled seq not added to running until after schedule returns
        # (it's in fresh_running, appended after Phase 2)

    def test_budget_exhaustion_by_prefill(self):
        """Small budget, Phase 1 consumes all -> Phase 2 runs zero decode."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        # Waiting seq exactly fills budget
        seq_waiting = make_seq(list(range(8)))
        seq_running = make_seq(list(range(100, 105)))
        bm.allocate(seq_running, num_cached_blocks=0)
        seq_running.num_cached_tokens = 5
        seq_running.status = SequenceStatus.RUNNING
        seq_running.is_prefill = False

        # Budget=8 means prefill uses all, no room for decode
        config = make_config(num_blocks, max_batched=8)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_running]))
        sched.add(seq_waiting)

        scheduled, has_prefill = sched.schedule()

        assert seq_waiting in scheduled
        assert seq_running not in scheduled
        assert has_prefill is True

    def test_mixed_prefill_completion(self):
        """Prefill completes + decode scheduled same step -> postprocess correct."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        # Waiting seq: exactly fills one block's worth (4 tokens at BS=4)
        seq_waiting = make_seq(list(range(4)))
        seq_running = make_seq(list(range(100, 105)))
        bm.allocate(seq_running, num_cached_blocks=0)
        seq_running.num_cached_tokens = 5
        seq_running.status = SequenceStatus.RUNNING
        seq_running.is_prefill = False

        config = make_config(num_blocks)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_running]))
        sched.add(seq_waiting)

        scheduled, has_prefill = sched.schedule()

        # Both scheduled: seq_waiting completes prefill, seq_running decodes
        assert seq_waiting in scheduled
        assert seq_running in scheduled
        assert seq_waiting.is_prefill is True
        assert seq_running.is_prefill is False

        # Simulate postprocess: prefill seq gets token 0, decode seq gets next token
        token_ids = [42, 99]  # one per seq
        sched.postprocess(scheduled, token_ids)

        # Prefill seq: completed prefill -> token appended
        assert seq_waiting.num_cached_tokens == 4
        assert seq_waiting.num_scheduled_tokens == 0
        assert seq_waiting.num_completion_tokens == 1
        assert seq_waiting.last_token == 42

        # Decode seq: token appended
        assert seq_running.num_completion_tokens == 1
        assert seq_running.last_token == 99

    def test_chunked_prefill_with_decode(self):
        """Large prefill chunked across steps, decode mixed into each step."""
        self.setup_method()
        num_blocks = 50
        bm = BlockManager(num_blocks, BS)

        # Large waiting seq: 30 tokens. Budget=10 means 3 chunks.
        seq_large = make_seq(list(range(30)))

        seq_running = make_seq(list(range(100, 105)))
        bm.allocate(seq_running, num_cached_blocks=0)
        seq_running.num_cached_tokens = 5
        seq_running.status = SequenceStatus.RUNNING
        seq_running.is_prefill = False

        config = make_config(num_blocks, max_batched=10)
        sched = make_scheduler(config, bm,
                               waiting=[],
                               running=deque([seq_running]))
        sched.add(seq_large)

        # Step 1: First chunk (10 tokens) + possible decode
        scheduled, has_prefill = sched.schedule()
        assert seq_large in scheduled
        assert seq_large.num_scheduled_tokens == 10
        assert seq_large.is_prefill is True

        # Simulate postprocess (chunked prefill: no token generation yet)
        sched.postprocess(scheduled, [99])
        assert seq_large.num_cached_tokens == 10
        assert seq_large.num_completion_tokens == 0  # mid-prefill, no token
        assert seq_large.status == SequenceStatus.WAITING  # not done

        # Step 2: Second chunk + decode
        scheduled, has_prefill = sched.schedule()
        assert seq_large.num_scheduled_tokens == 10
        sched.postprocess(scheduled, [99])
        assert seq_large.num_cached_tokens == 20

        # Step 3: Final chunk (10 tokens) completes prefill
        scheduled, has_prefill = sched.schedule()
        assert seq_large.num_scheduled_tokens == 10
        sched.postprocess(scheduled, [42])
        assert seq_large.num_cached_tokens == 30
        assert seq_large.num_completion_tokens == 1  # prefill complete, token generated
        assert seq_large.last_token == 42

    def test_fresh_seq_not_in_phase2(self):
        """Seq completing prefill in Phase 1 is NOT scheduled for decode in same step."""
        self.setup_method()
        num_blocks = 10
        bm = BlockManager(num_blocks, BS)

        # Waiting seq: 4 tokens = 1 block (completes prefill in one step)
        seq_waiting = make_seq(list(range(4)))

        config = make_config(num_blocks)
        sched = make_scheduler(config, bm)
        sched.add(seq_waiting)

        scheduled, has_prefill = sched.schedule()

        # Seq appears exactly once in scheduled_seqs
        assert scheduled.count(seq_waiting) == 1
        assert seq_waiting.is_prefill is True
        # Seq is in fresh_running, not directly added to self.running during Phase 1

    def test_mid_prefill_not_appended_to_running(self):
        """Mid-prefill seq stays in waiting, not added to running prematurely."""
        self.setup_method()
        num_blocks = 50
        bm = BlockManager(num_blocks, BS)

        # Large seq that gets chunked: 20 tokens, budget=5 -> 4 chunks
        seq = make_seq(list(range(20)))

        config = make_config(num_blocks, max_batched=5)
        sched = make_scheduler(config, bm)
        sched.add(seq)

        # First chunk: 5 tokens, mid-prefill
        scheduled, has_prefill = sched.schedule()
        assert seq.num_scheduled_tokens == 5
        assert seq not in sched.running  # mid-prefill, in waiting heap
        assert seq.status == SequenceStatus.WAITING


print("Running mixed scheduler tests...")
import traceback


def run(cls):
    print(f"--- {cls.__name__} ---")
    inst = cls()
    for name in sorted(dir(inst)):
        if name.startswith("test_"):
            try:
                inst.setup_method()
                getattr(inst, name)()
                print(f"  PASS {name}")
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                traceback.print_exc()


run(TestMixedBatchBasic)

print("Done.")
