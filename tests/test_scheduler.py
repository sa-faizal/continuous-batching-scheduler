from __future__ import annotations

import unittest

from mini_scheduler import RequestState, Scheduler
from mini_scheduler.model import next_token


def expected_tokens(
    request_id: str, prompt: tuple[str, ...], count: int
) -> list[str]:
    context = list(prompt)
    result: list[str] = []
    for position in range(count):
        token = next_token(request_id, position, tuple(context))
        result.append(token)
        context.append(token)
    return result


class SchedulerBaselineTests(unittest.TestCase):
    def test_prefill_decode_and_completion(self) -> None:
        scheduler = Scheduler(num_blocks=1, max_batch_size=2)
        request = scheduler.submit("a", ["hello"], max_new_tokens=3)

        scheduler.run_until_idle()

        self.assertEqual(RequestState.COMPLETED, request.state)
        self.assertEqual(expected_tokens("a", ("hello",), 3), request.generated)
        self.assertEqual(1, scheduler.cache.available)
        scheduler.assert_invariants()

    def test_continuous_batching_shares_a_completed_prefix(self) -> None:
        scheduler = Scheduler(num_blocks=1, max_batch_size=4)
        first = scheduler.submit("first", ["shared"], max_new_tokens=2)

        scheduler.step()  # Dispatch first's prefill.
        second = scheduler.submit("second", ["shared"], max_new_tokens=2)
        scheduler.step()  # Retire prefill, share its block, batch both decodes.

        self.assertEqual(first.block_ids, second.block_ids)
        self.assertEqual(2, len(scheduler.inflight))

        scheduler.run_until_idle()
        self.assertEqual(RequestState.COMPLETED, first.state)
        self.assertEqual(RequestState.COMPLETED, second.state)
        self.assertEqual(
            expected_tokens("first", ("shared",), 2), first.generated
        )
        self.assertEqual(
            expected_tokens("second", ("shared",), 2), second.generated
        )
        self.assertEqual(1, scheduler.cache.available)
        scheduler.assert_invariants()

    def test_cancel_while_queued_is_idempotent(self) -> None:
        scheduler = Scheduler(num_blocks=1)
        request = scheduler.submit("queued", ["prompt"], max_new_tokens=1)

        self.assertTrue(scheduler.cancel("queued"))
        self.assertFalse(scheduler.cancel("queued"))
        scheduler.run_until_idle()

        self.assertEqual(RequestState.CANCELLED, request.state)
        self.assertEqual([], request.generated)
        scheduler.assert_invariants()

    def test_duplicate_request_id_is_rejected(self) -> None:
        scheduler = Scheduler(num_blocks=1)
        scheduler.submit("same", ["one"], max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "duplicate request id"):
            scheduler.submit("same", ["two"], max_new_tokens=1)


class CancellationRegressionTest(unittest.TestCase):
    def test_cancel_does_not_release_another_sequences_shared_block(self) -> None:
        scheduler = Scheduler(num_blocks=1, max_batch_size=3)
        cancelled = scheduler.submit("cancelled", ["shared"], max_new_tokens=2)

        scheduler.step()  # Dispatch the shared prefix prefill.
        survivor = scheduler.submit("survivor", ["shared"], max_new_tokens=2)
        scheduler.step()  # Share the block and dispatch both first decode tokens.

        self.assertEqual(cancelled.block_ids, survivor.block_ids)
        self.assertEqual(2, scheduler.cache.blocks[0].ref_count)

        scheduler.cancel("cancelled")
        newcomer = scheduler.submit("newcomer", ["other"], max_new_tokens=1)
        scheduler.step()  # Retire the shared in-flight batch.

        # The survivor must retain its context and ownership until completion. In
        # the starter, newcomer reuses the block and survivor dispatches its next
        # decode from a context that has lost the shared prompt.
        survivor_work = next(
            work for work in scheduler.inflight if work.request_id == "survivor"
        )
        self.assertEqual(
            expected_tokens("survivor", ("shared",), 2)[1],
            survivor_work.payload[0],
        )
        self.assertEqual(RequestState.CANCELLED, cancelled.state)
        self.assertEqual(RequestState.QUEUED, newcomer.state)
        scheduler.assert_invariants()


if __name__ == "__main__":
    unittest.main()
