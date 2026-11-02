# Per-layer expert paging runtime harness

`PagingRuntime` is a synchronous adapter between the existing `ExpertSource`
and `SlotCache`. It handles one MoE layer's selected expert IDs at a time. It
does not add a scheduler, choose router IDs, allocate GPU banks, or call Vulkan.
The future llama.cpp adapter remains responsible for locating the slot-map
copy callback, building actual cache tensors, and supplying a GPU-use
completion token.

## Construction and metadata

Create one `PagingQuarantine` with an explicit staging capacity, then create a
runtime with the cache, validated source, upload backend and a gate/up/down
mapping for each supported layer. The runtime copies the layer map and exposes
no mutation API. Each tensor reference carries its source tensor ID and exact
per-expert byte stride from the same validated metadata used to open
`ExpertSource`. `read_expert()` reports success but does not return a byte
count, so the adapter cannot discover a mismatched metadata stride by itself.
The caller must build these references from the validated descriptor table.

The per-buffer staging capacity must be nonzero, no larger than
`ExpertSource::max_expert_bytes()`, and large enough for every mapped role.
`PagingQuarantine(capacity)` keeps the existing one-buffer allocation. To opt
into batching, pass an explicit second argument from 2 through
`kHardMaxUploadBatchCapacity` (currently 8). The pool then allocates exactly
`capacity * batch_capacity` bytes as fixed reusable slices. The caller must
account for that full physical-memory cost before opting in; the runtime does
not grow the pool automatically. Zero, over-limit, and overflowing pool sizes
are invalid. The raw selected-expert request is bounded (64 by default, 256
maximum); repeated IDs are deduplicated by `SlotCache` and consume only one set
of reads and slots.

## Per-layer transaction

`load_layer(layer, selected_ids, gpu_use_token)` performs these steps:

1. Check configuration, cancellation, layer mapping and request bounds.
2. Atomically reserve every unique `(layer, expert)` slot before reading.
3. For each missing expert, read gate, up and down bytes in sequence into
   distinct staging slices. A successful `read_expert()` has written the full
   requested role stride, so the runtime does not clear that range first. A
   failed read is never uploaded. Submit up to the configured batch capacity
   before waiting for fences, then wait for every accepted transfer in that
   batch before reusing any slice.
4. Mark an expert complete only after all three of its role transfers report
   `Complete`. A read, submit, cancellation, or quiesced fence failure drains
   all previously accepted transfers before the layer reservation is rolled
   back.
5. Copy the result assignments before commit, then atomically commit all keys
   against the supplied future GPU-use token. The caller may submit the MoE
   compute only after `load_layer()` returns `Ok`; call `SlotCache::complete()`
   only after that compute's GPU completion signal.

Cache hits skip reads/uploads but are also pinned by the committed token.
Commit failure rolls back its still-active reservation. The adapter preserves
the structured source error and cache status in `PagingResult`.

Cancellation is checked before reads, between role reads/uploads and before
commit. A cancellation during `wait_transfer()` cannot interrupt that fence;
the adapter still drains every accepted transfer. It rolls back only after all
fences prove quiescence. If any fence is indeterminate, it quarantines the
whole staging pool and active reservation. Cancellation must be serialized
with the final commit/use handoff: if the caller cancels after an `Ok` result
but before submitting compute, it must release that unused token only when it
can guarantee no GPU work was submitted.

## Transfer and quarantine contract

`submit_upload()` returning `submitted=false` guarantees no work was accepted
and no source pointer is retained. A submitted operation returns a transfer
token, and the backend may have up to the configured batch capacity of
outstanding uploads. The runtime passes their tokens to `wait_transfers()` in
submission order. Its default implementation calls `wait_transfer()` for each
token; a backend may override it with aggregate synchronization only when it
can fill a correct per-token status for every submitted token. `wait_transfer()`
may report:

- `Complete`: transfer finished; staging can be reused.
- `FailedButQuiesced`: transfer failed but no longer accesses staging or slot;
  the cache transaction can safely roll back.
- `Indeterminate`: the backend cannot prove the transfer is finished.

After a read failure, submission rejection, cancellation, or a failed fence,
the runtime waits for every other accepted transfer before returning or
rolling back. It keeps waiting after an `Indeterminate` result so every fence
that can be resolved is resolved. If any transfer is indeterminate (including
a submission without a usable token), the runtime is poisoned. It moves the
active reservation into `PagingQuarantine`, which retains the entire staging
pool and records every unresolved transfer. Further runtime work returns
`Quarantined`. The quarantine is bound to the same backend object.

After stopping/draining that backend, call
`release_after_backend_quiesced(backend)`. It calls the backend-wide
`quiesce()` method and rolls back only when that method returns
`Complete` or `FailedButQuiesced`. An indeterminate result leaves every staging
slice and the slot reservation quarantined. Destroying a quarantine while it
still owns such a reservation terminates the process rather than silently
making unsafe memory reusable; keep the quarantine alive until quiescence is
established. The cache, source, backend and quarantine must outlive an active
runtime/reservation.

This makes uncertain-fence recovery deliberately explicit. It does not cancel
a blocking backend wait, and a backend that cannot provide either a per-upload
completion or a backend-wide quiescence guarantee cannot safely use this
adapter.

## Host validation

From the repository root:

```sh
c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic \
    ps5/moe/expert_source.cpp ps5/moe/slot_cache.cpp \
    ps5/moe/paging_runtime.cpp ps5/moe/paging_runtime_test.cpp \
    -o /tmp/ps5lm_paging_runtime_test
/tmp/ps5lm_paging_runtime_test
```

The test uses real `ExpertSource` reads from a deterministic tiny fixture and
a fake backend that defers copying staging bytes until its fence. It checks
default single-buffer behavior, explicit bounded batches, distinct source
bytes and submit-before-wait ordering, default per-token and overridable batch
fence paths, partial submit rejection, cancellation and failed-fence draining,
early, late and aggregate indeterminate fences, cache reservation and GPU pin
lifetimes, invalid completion tokens, and failed then successful backend-wide
quiescence. No model, title, scheduler, thread pool, or Vulkan backend is used.
