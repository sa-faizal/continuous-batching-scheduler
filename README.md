# Continuous-batching scheduler corruption

## Problem statement

Think of a KV-cache block as a reusable piece of memory that a request checks out
while generating tokens. Requests with the same prompt may share a block, and the
scheduler returns blocks to a small free pool when they are no longer needed.

The difficult case is cancellation. A request can be cancelled while previously
submitted GPU work is still finishing. If that request shares a block with another
request, returning the block too early lets a new request reuse memory that is
still in use. The surviving request may then read another request's context,
receive incorrect tokens, leak a block, or decrement its reference count twice.
Even after the reference-count bug is fixed, late work must be distinguishable
from work targeting a newer allocation of the same physical block.

This repository contains a small, deterministic model of that failure. No GPU or
third-party package is required. The scheduler supports:

- prefill and decode work;
- a fixed-size KV-cache block pool;
- continuous batching;
- request cancellation; and
- exact-prefix cache sharing.

## Task

The exercise is time-boxed to approximately 45 minutes. Your goal is to make
cancellation safe without slowing the scheduler down or turning off prefix
sharing.

### 1. See the failure

Run the test suite. Four tests pass and two fail. The first failure follows three
requests:

- one request creates a cached prefix;
- a second request shares that prefix;
- the first request is cancelled while decode work is still outstanding; and
- a third request creates pressure to reuse the only available block.

Trace the request states, reference count, GPU leases, and free-list membership at
each `step()`. Explain exactly when the scheduler first considers a live block
reusable.

### 2. Fix the bug

Change the implementation so that:

- accepting a cancellation does not imply that submitted GPU work has stopped;
- a block is reusable only after no request and no outstanding work still needs
  it;
- cancellation and normal completion release each owned block exactly once; and
- all existing tests pass.

You may change request states, APIs, and internal data structures. Be prepared to
explain the ownership rule your implementation relies on.

### 3. Reject stale work after block reuse

The second failing test models an ABA problem:

1. Physical block `0` is allocated to one sequence.
2. It is released and then allocated again to another sequence.
3. Late work from the first allocation still refers to block `0`.

Add an allocation generation, epoch, or equivalent identity so stale work is
rejected instead of modifying the new owner's block. Reusing the physical block ID
is expected; confusing two different lifetimes of that block is not.

### 4. Preserve the serving behavior

Keep continuous batching and prefix-cache sharing. Do not solve the problem by
serializing all requests or by holding one global lock across scheduling,
admission, and completion.

### 5. Add one focused test

Add at least one meaningful test beyond the existing regression. Good choices
include:

- cancellation during prefill or decode;
- repeated cancellation;
- normal completion followed by immediate block reuse; or
- cancellation while another request shares the prefix.

The test should reproduce a deliberate sequence of scheduler steps rather than
depend on timing or sleeps.

### 6. Discuss the design

Be ready to explain:

- the request states and legal cancellation transitions;
- the difference between request lifetime, outstanding GPU work, and physical
  block lifetime;
- how cancellation can enter the scheduler without a global lock; and
- how a production service should react if it detects possible KV corruption.

Writing a polished `DESIGN.md` is not required during the time box. It is provided
as optional scratch space.

### Definition of done

- The complete test suite passes.
- At least one focused regression test is added.
- Continuous batching and prefix sharing still work.
- Cleanup happens exactly once and a live block is not reused.
- Late work cannot mutate a newer allocation of the same physical block.

### Stretch goals

If time remains, consider adding explicit runtime invariants, broader cancellation
coverage, or notes in `DESIGN.md`. These are useful discussion topics but are not
required for completion.

## Running the exercise

```text
python -m unittest discover -s tests -v
```

The starter intentionally has two failing regression tests. The other tests
provide a behavioral baseline. A complete solution should make the entire suite
pass and should add meaningful coverage of its own.

Start with [`mini_scheduler/scheduler.py`](mini_scheduler/scheduler.py), then inspect
[`mini_scheduler/kv_cache.py`](mini_scheduler/kv_cache.py). The simulated GPU has a
one-tick completion delay: `step()` first retires the previous batch, admits queued
requests, and then dispatches the next batch.

## Scope

The model deliberately uses one immutable physical prompt block per unique active
prompt, keeps each request's generated tail in logical state, and supports only
exact-prompt prefix hits. Those simplifications keep the exercise focused on
lifecycle and ownership rather than token packing. Production considerations such
as multi-block sequences, copy-on-write tail blocks, partial-prefix matching,
distributed workers, and real CUDA/ROCm events belong in the design discussion,
not necessarily in the implementation.
