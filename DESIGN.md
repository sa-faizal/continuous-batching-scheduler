# Optional design notes

This file is optional scratch space for design discussion. A polished written
response is not required during the time-boxed exercise.

## State machine

Describe the legal request states and transitions, including cancellation while
GPU work is outstanding.

## Ownership and lifetime

Describe logical request ownership, GPU work/leases, prefix-cache references, and
the condition under which a physical block may return to the free list.

## Concurrency and throughput

Describe how cancellation and GPU completions are coordinated without a global
scheduler lock, and why continuous-batching throughput is preserved.

## Invariants and production recovery

List the invariants enforced by the implementation. Explain how a production
system should contain, diagnose, and recover from a violation without returning
silently corrupted output.
