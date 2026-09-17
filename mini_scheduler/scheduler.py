"""A miniature continuous-batching scheduler with an intentional lifecycle bug."""

from __future__ import annotations

from collections import Counter, deque

from .kv_cache import KVCache
from .model import Request, RequestState, TokenEvent, WorkItem, next_token


class InvariantViolation(RuntimeError):
    pass


class Scheduler:
    """Single-coordinator scheduler with one-tick simulated GPU work.

    The implementation is intentionally incomplete around cancellation ownership.
    It is the subject of the exercise and is not production-ready code.
    """

    def __init__(self, num_blocks: int, max_batch_size: int = 4) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.cache = KVCache(num_blocks)
        self.max_batch_size = max_batch_size
        self.requests: dict[str, Request] = {}
        self._queued: deque[str] = deque()
        self._inflight: list[WorkItem] = []
        self._outputs: list[TokenEvent] = []
        # Entries are weak aliases: active requests, not the cache, own references.
        self._prefix_cache: dict[tuple[str, ...], int] = {}

    @property
    def has_work(self) -> bool:
        return bool(self._queued or self._inflight) or any(
            not request.is_terminal for request in self.requests.values()
        )

    @property
    def inflight(self) -> tuple[WorkItem, ...]:
        return tuple(self._inflight)

    def submit(
        self, request_id: str, prompt: list[str] | tuple[str, ...], max_new_tokens: int
    ) -> Request:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id: {request_id}")
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        request = Request(request_id, tuple(prompt), max_new_tokens)
        self.requests[request_id] = request
        self._queued.append(request_id)
        return request

    def cancel(self, request_id: str) -> bool:
        request = self.requests.get(request_id)
        if request is None or request.is_terminal:
            return False

        request.state = RequestState.CANCELLED
        # Cancellation eagerly returns logical ownership. Outstanding work retires
        # independently and performs its normal terminal cleanup.
        self._release_request_blocks(request)
        return True

    def step(self) -> None:
        """Retire one GPU batch, admit waiters, and dispatch another batch."""

        self._retire_batch()
        self._admit_queued()
        self._dispatch_batch()

    def run_until_idle(self, max_steps: int = 100) -> None:
        for _ in range(max_steps):
            if not self.has_work:
                return
            self.step()
        raise TimeoutError(f"scheduler did not become idle after {max_steps} steps")

    def poll_outputs(self) -> list[TokenEvent]:
        outputs, self._outputs = self._outputs, []
        return outputs

    def assert_invariants(self) -> None:
        """Check cross-component ownership and physical-pool invariants."""

        expected_refs: Counter[int] = Counter()
        for request in self.requests.values():
            if not request.is_terminal:
                expected_refs.update(request.block_ids)

        expected_leases: Counter[int] = Counter(
            work.block_id for work in self._inflight
        )
        free_ids = self.cache.free_ids
        if len(free_ids) != len(set(free_ids)):
            raise InvariantViolation("free list contains a block more than once")

        for block in self.cache.blocks:
            if block.ref_count != expected_refs[block.block_id]:
                raise InvariantViolation(
                    f"block {block.block_id}: physical refs={block.ref_count}, "
                    f"logical refs={expected_refs[block.block_id]}"
                )
            if block.gpu_leases != expected_leases[block.block_id]:
                raise InvariantViolation(
                    f"block {block.block_id}: leases={block.gpu_leases}, "
                    f"in-flight work={expected_leases[block.block_id]}"
                )
            should_be_free = block.ref_count == 0 and block.gpu_leases == 0
            if block.on_free_list != should_be_free:
                raise InvariantViolation(
                    f"block {block.block_id}: free-list state disagrees with owners"
                )
            if block.on_free_list != (block.block_id in free_ids):
                raise InvariantViolation(
                    f"block {block.block_id}: free-list membership mismatch"
                )

        for prompt, block_id in self._prefix_cache.items():
            block = self.cache.blocks[block_id]
            if block.on_free_list or tuple(block.tokens[: len(prompt)]) != prompt:
                raise InvariantViolation(
                    f"prefix entry {prompt!r} points at invalid block {block_id}"
                )

    def _retire_batch(self) -> None:
        batch, self._inflight = self._inflight, []
        for work in batch:
            request = self.requests[work.request_id]
            request.in_flight = False

            if request.state is RequestState.CANCELLED:
                self.cache.release_gpu_lease(work.block_id)
                self._release_request_blocks(request)
                continue

            if work.phase == "prefill":
                self.cache.replace(work.block_id, work.payload)
                self._prefix_cache[request.prompt] = work.block_id
                request.state = RequestState.DECODING
            else:
                token = work.payload[0]
                request.generated.append(token)
                self._outputs.append(TokenEvent(request.request_id, token))

            self.cache.release_gpu_lease(work.block_id)
            if len(request.generated) == request.max_new_tokens:
                request.state = RequestState.COMPLETED
                self._release_request_blocks(request)

    def _admit_queued(self) -> None:
        waiting = len(self._queued)
        for _ in range(waiting):
            request_id = self._queued.popleft()
            request = self.requests[request_id]
            if request.state is RequestState.CANCELLED:
                continue

            shared_block = self._prefix_cache.get(request.prompt)
            if shared_block is not None:
                block = self.cache.blocks[shared_block]
                if not block.on_free_list and block.ref_count > 0:
                    self.cache.acquire(shared_block)
                    request.block_ids.append(shared_block)
                    request.state = RequestState.DECODING
                    continue
                del self._prefix_cache[request.prompt]

            block_id = self.cache.allocate()
            if block_id is None:
                self._queued.append(request_id)
                continue
            request.block_ids.append(block_id)
            request.state = RequestState.PREFILL

    def _dispatch_batch(self) -> None:
        candidates = [
            request
            for request in self.requests.values()
            if request.state in {RequestState.PREFILL, RequestState.DECODING}
            and not request.in_flight
        ]
        # Prefills first is a simple, deterministic policy for this exercise.
        candidates.sort(key=lambda request: request.state is RequestState.DECODING)

        for request in candidates[: self.max_batch_size]:
            block_id = request.block_ids[-1]
            if request.state is RequestState.PREFILL:
                work = WorkItem(request.request_id, "prefill", block_id, request.prompt)
            else:
                token = next_token(
                    request.request_id,
                    len(request.generated),
                    self.cache.read(block_id) + tuple(request.generated),
                )
                work = WorkItem(request.request_id, "decode", block_id, (token,))
            self.cache.acquire_gpu_lease(block_id)
            request.in_flight = True
            self._inflight.append(work)

    def _release_request_blocks(self, request: Request) -> None:
        for block_id in request.block_ids:
            self.cache.release(block_id)
            block = self.cache.blocks[block_id]
            if block.ref_count == 0:
                stale = [
                    prompt
                    for prompt, cached_id in self._prefix_cache.items()
                    if cached_id == block_id
                ]
                for prompt in stale:
                    del self._prefix_cache[prompt]
