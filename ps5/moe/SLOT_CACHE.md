# Transactional MoE slot policy

`SlotCache` is a backend-independent ownership policy for a fixed number of
expert-weight slots. It stores only bounded metadata; it does not allocate
weight buffers, perform file reads, upload data, submit GPU work, or call
Vulkan. A later adapter owns those operations and supplies the configured slot
count, bytes per slot, and byte budget.

The policy preserves the complete `(layer, expert)` identity. A request is
deduplicated by that identity, so repeated routing IDs share one slot
assignment. Construction rejects zero or overflowing capacities, a payload
capacity above the caller's budget, more than 4096 slots/request keys, and
more than 256 in-flight completion tokens. Request, reservation, token, and
slot metadata allocations have those fixed bounds; they do not scale with
model weight size. `metadata_bytes` reports only the fixed slot descriptor
array, not allocator overhead or the separately bounded request/token vectors.

## Transaction lifecycle

1. Call `reserve(keys)`. It plans the whole unique key set under one lock. It
   protects resident hits and chooses empty slots before LRU residents with no
   reservation and no outstanding GPU-use pin. It changes nothing if it cannot
   reserve the complete set. A key already reserved by another transaction
   rejects the whole request with `KeyReserved`.
2. For each `needs_upload` assignment, read and upload the expert into the
   assigned slot. Call `mark_upload_complete(key)` only after both have
   succeeded. A failed read/upload or cancelled request must call `rollback()`;
   the reservation destructor also rolls it back.
3. Once every upload is complete and the caller has a unique, increasing
   completion token for the GPU use, call `commit(token)`. The operation
   publishes all newly filled identities together and pins every slot in the
   batch to that token. A missing upload or token-capacity error leaves the
   reservation active for retry or rollback.
4. After the backend confirms GPU completion for that token, call
   `complete(token)`. Only this releases the token's slot pins. Unknown or
   repeated token completion is rejected.

An eviction reservation removes the old identity from lookup immediately,
because the adapter may overwrite that payload while processing the request.
Rolling the reservation back frees that slot; the old identity is not restored
because the policy cannot know whether an upload partially overwrote it. This
may cause a later cache miss, but cannot expose failed or partial payload as a
resident expert. Resident hits remain valid on rollback.

The cache must outlive every `Reservation` handle. Each handle is intended to
be operated by one caller thread; separate reservations and cache queries are
mutex-protected. The adapter must retain committed slots until each associated
GPU completion token signals. For a cancelled operation that already submitted
GPU work, the adapter must first ensure that work cannot still access an
uncommitted/reused slot; a CPU cancellation flag alone is not a GPU fence.

## Host validation

From the repository root:

```sh
c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic \
    ps5/moe/slot_cache.cpp ps5/moe/slot_cache_test.cpp \
    -o /tmp/ps5lm_slot_cache_test
/tmp/ps5lm_slot_cache_test
```

The deterministic host suite uses a separate LRU oracle for slot selection
and direct adversarial cases for repeated IDs, concurrent reservations,
capacity and budget rejection, upload/read rollback, cancellation, token
ordering/limits, overlapping pins, and eviction before and after fence
completion. It does not validate a title adapter or GPU synchronization.
