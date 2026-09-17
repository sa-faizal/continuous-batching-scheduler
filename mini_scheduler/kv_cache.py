"""A deliberately small physical KV-block pool."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class KVBlock:
    block_id: int
    tokens: list[str] = field(default_factory=list)
    ref_count: int = 0
    gpu_leases: int = 0
    on_free_list: bool = True


class KVCache:
    """Tracks logical references separately from outstanding GPU leases.

    A block is reusable only when both counters reach zero. The class intentionally
    does not know which request owns a reference; cross-component ownership is a
    scheduler invariant.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def available(self) -> int:
        return len(self._free)

    @property
    def free_ids(self) -> tuple[int, ...]:
        return tuple(self._free)

    def allocate(self) -> int | None:
        if not self._free:
            return None
        block_id = self._free.popleft()
        block = self.blocks[block_id]
        if not block.on_free_list or block.ref_count or block.gpu_leases:
            raise RuntimeError(f"free-list corruption for block {block_id}")
        block.on_free_list = False
        block.ref_count = 1
        block.tokens.clear()
        return block_id

    def acquire(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.on_free_list or block.ref_count < 1:
            raise RuntimeError(f"cannot acquire free block {block_id}")
        block.ref_count += 1

    def release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count < 1:
            raise RuntimeError(f"reference-count underflow for block {block_id}")
        block.ref_count -= 1
        self._maybe_free(block)

    def acquire_gpu_lease(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.on_free_list:
            raise RuntimeError(f"cannot lease free block {block_id}")
        block.gpu_leases += 1

    def release_gpu_lease(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.gpu_leases < 1:
            raise RuntimeError(f"GPU-lease underflow for block {block_id}")
        block.gpu_leases -= 1
        self._maybe_free(block)

    def read(self, block_id: int) -> tuple[str, ...]:
        return tuple(self.blocks[block_id].tokens)

    def replace(self, block_id: int, tokens: tuple[str, ...]) -> None:
        self.blocks[block_id].tokens[:] = tokens

    def append(self, block_id: int, token: str) -> None:
        self.blocks[block_id].tokens.append(token)

    def _maybe_free(self, block: KVBlock) -> None:
        if block.ref_count == 0 and block.gpu_leases == 0:
            if block.on_free_list:
                raise RuntimeError(f"block {block.block_id} entered free list twice")
            block.tokens.clear()
            block.on_free_list = True
            self._free.append(block.block_id)

