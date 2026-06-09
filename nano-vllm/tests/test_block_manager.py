from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BS = 4  # block_size for tests


def make_seq(token_ids):
    return Sequence(token_ids)


class TestFreeTailBlocks:

    def setup_method(self):
        Sequence.block_size = BS

    def test_free_one_from_multi_block_seq(self):
        self.setup_method()
        bm = BlockManager(num_blocks=8, block_size=BS)
        seq = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 9 tokens → 3 blocks
        bm.allocate(seq, num_cached_blocks=0)
        seq.num_cached_tokens = 9  # simulate prefill done

        freed_block = seq.block_table[-1]
        assert len(seq.block_table) == 3

        bm.free_tail_blocks(seq, 1)

        assert len(seq.block_table) == 2
        assert seq.num_cached_tokens == 8  # 2 * BS

        assert bm.blocks[freed_block].ref_count == 0
        assert freed_block in bm.free_block_ids
        assert freed_block not in bm.used_block_ids
        for bid in seq.block_table:
            assert bid in bm.used_block_ids

    def test_free_all_equivalent_to_deallocate(self):
        self.setup_method()
        bm = BlockManager(num_blocks=8, block_size=BS)
        seq = make_seq([1, 2, 3, 4, 5])  # 5 tokens → 2 blocks
        bm.allocate(seq, num_cached_blocks=0)
        seq.num_cached_tokens = 5
        block_ids = list(seq.block_table)

        bm.free_tail_blocks(seq, 2)

        assert len(seq.block_table) == 0
        assert seq.num_cached_tokens == 0
        for bid in block_ids:
            assert bm.blocks[bid].ref_count == 0
            assert bid in bm.free_block_ids
            assert bid not in bm.used_block_ids

    def test_shared_block_not_deallocated(self):
        """Block shared via ref_count > 1 should not be freed if tail blocks
        don't include it."""
        self.setup_method()
        bm = BlockManager(num_blocks=8, block_size=BS)

        seq_a = make_seq(list(range(12)))  # 12 tokens → 3 blocks
        bm.allocate(seq_a, num_cached_blocks=0)
        seq_a.num_cached_tokens = 12

        # Simulate prefix sharing: bump ref_count on block 0
        shared = seq_a.block_table[0]
        bm.blocks[shared].ref_count += 1

        # Free only blocks 1 and 2
        bm.free_tail_blocks(seq_a, 2)

        # Block 0 (shared) should NOT be deallocated
        assert bm.blocks[shared].ref_count >= 1
        assert shared in bm.used_block_ids

        # Blocks 1, 2 should be freed
        assert len(seq_a.block_table) == 1
        assert seq_a.num_cached_tokens == BS  # 1 * BS

    def test_free_from_single_block_seq(self):
        self.setup_method()
        bm = BlockManager(num_blocks=8, block_size=BS)
        seq = make_seq([1, 2, 3])  # 3 tokens → 1 block
        bm.allocate(seq, num_cached_blocks=0)
        seq.num_cached_tokens = 3
        block_id = seq.block_table[0]

        bm.free_tail_blocks(seq, 1)

        assert len(seq.block_table) == 0
        assert seq.num_cached_tokens == 0
        assert bm.blocks[block_id].ref_count == 0
        assert block_id in bm.free_block_ids
