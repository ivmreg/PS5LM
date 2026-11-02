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

The staging capacity must be nonzero, no larger than
`ExpertSource::max_expert_bytes()`, and large enough for every mapped role.
`PagingQuarantine` owns exactly one fixed-size byte vector. The runtime uses it
for each gate, up and down read in turn. The raw selected-expert request is
bounded (64 by default, 256 maximum); repeated IDs are deduplicated by
`SlotCache` and consume only one set of reads and slots.

## Per-layer transaction

`load_layer(layer, selected_ids, gpu_use_token)` performs these steps:

1. Check configuration, cancellation, layer mapping and request bounds.
2. Atomically reserve every unique `(layer, expert)` slot before reading.
3. For each missing expert, read gate, up and down bytes in sequence into the
   single staging buffer. Submit one upload and wait for its completion before
   reading into that buffer again.
4. Mark that expert complete only after all three role uploads finish. A
   read/upload/cancellation error rolls the entire layer reservation back.
5. Copy the result assignments before commit, then atomically commit all keys
   against the supplied future GPU-use token. The caller may submit the MoE
   compute only after `load_layer()` returns `Ok`; call `SlotCache::complete()`
   only after that compute's GPU completion signal.

Cache hits skip reads/uploads but are also pinned by the committed token.
Commit failure rolls back its still-active reservation. The adapter preserves
the structured source error and cache status in `PagingResult`.

Cancellation is checked before reads, between role reads/uploads and before
commit. A cancellation during `wait_transfer()` cannot interrupt that fence;
the adapter waits for a safe result, then rolls back. Cancellation must be
serialized with the final commit/use handoff: if the caller cancels after an
`Ok` result but before submitting compute, it must release that unused token
only when it can guarantee no GPU work was submitted.

## Transfer and quarantine contract

`submit_upload()` returning `submitted=false` guarantees no work was accepted
and no source pointer is retained. A submitted operation returns a transfer
token. `wait_transfer()` may report:

- `Complete`: transfer finished; staging can be reused.
- `FailedButQuiesced`: transfer failed but no longer accesses staging or slot;
  the cache transaction can safely roll back.
- `Indeterminate`: the backend cannot prove the transfer is finished.

On `Indeterminate`, the runtime is poisoned. It moves the active reservation
into `PagingQuarantine`, which retains the staging buffer and prevents the
reservation destructor from releasing/reusing its slot. Further runtime work
returns `Quarantined`. The quarantine is bound to the same backend object.

After stopping/draining that backend, call
`release_after_backend_quiesced(backend)`. It calls the backend-wide
`quiesce()` method and rolls back only when that method returns
`Complete` or `FailedButQuiesced`. An indeterminate result leaves both buffer
and slot quarantined. Destroying a quarantine while it still owns such a
reservation terminates the process rather than silently making unsafe memory
reusable; keep the quarantine alive until quiescence is established. The cache,
source, backend and quarantine must outlive an active runtime/reservation.

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
exact gate/up/down bytes, repeated IDs, layer mappings, cancellation, source
and upload errors, cache pressure/pins, invalid completion tokens and an
indeterminate fence followed by failed and successful backend-wide quiescence.
No model, title, scheduler or Vulkan backend is used.
