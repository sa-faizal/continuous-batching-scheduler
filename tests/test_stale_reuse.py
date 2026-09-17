from __future__ import annotations

import unittest

from mini_scheduler.kv_cache import KVCache


class StaleBlockReuseTest(unittest.TestCase):
    def test_old_allocation_cannot_modify_reused_physical_block(self) -> None:
        cache = KVCache(num_blocks=1)

        old_allocation = cache.allocate()
        self.assertIsNotNone(old_allocation)
        cache.release(old_allocation)

        new_allocation = cache.allocate()
        self.assertIsNotNone(new_allocation)

        # The physical block may be the same, but each allocation needs a distinct
        # identity so a late completion can be recognized as stale.
        self.assertNotEqual(old_allocation, new_allocation)
        with self.assertRaises(RuntimeError):
            cache.replace(old_allocation, ("late-write",))

        self.assertEqual((), cache.read(new_allocation))


if __name__ == "__main__":
    unittest.main()
