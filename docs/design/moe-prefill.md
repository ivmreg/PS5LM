# P5b: bounded MoE prefill and repeated requests

Status: source-backed feasibility design, 2026-11-01. No code changes.

## Scope and source pin

This design targets the Qwen3-MoE routed FFN path already present in the
repository. It assumes the P3 model loader has retained validated expert
descriptors without allocating expert payloads, P4 fills the existing cache
banks transactionally, and P5a provides an explicit scheduler-copy error and
GPU-completion path. It does not make a host build or PS5 timing claim.

The source trace below uses the repository's pinned llama.cpp commit
`04f7226a2c12c7f382c545a6db5a034266aa53cb` in `third_party/llama.cpp`. The
working submodule was clean while it was inspected. The existing P0 trace is
in [moe-paging.md](moe-paging.md).

## Current execution path and full-expert failure modes

### Context parameters and scheduler reservation

The public `llama_context_params` distinguishes `n_batch` (logical maximum
input tokens), `n_ubatch` (physical tokens per graph), and `n_seq_max` in
[`include/llama.h`](../../third_party/llama.cpp/include/llama.h#L365-L373).
Defaults are `n_batch=2048`, `n_ubatch=512`, and `n_seq_max=1` in
[`src/llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L3977-L3987).
Context construction clamps causal `n_batch` to `n_ctx` and then clamps
`n_ubatch` to `n_batch` ([same file](../../third_party/llama.cpp/src/llama-context.cpp#L332-L336)).
It creates the MoE cache, installs a new scheduler and calls `sched_reserve()`
during construction ([same file](../../third_party/llama.cpp/src/llama-context.cpp#L515-L556)).
So the first graph reservation occurs before the caller can submit a prompt.

`sched_new()` registers both the copy and reallocation callbacks
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L707-L713)).
`sched_reserve()` synchronizes, then reserves a prompt-processing graph of
`min(n_ctx,n_ubatch)` tokens, a token-generation graph of `n_seq_max` tokens,
and the prompt graph again to reduce later reallocations
([same file](../../third_party/llama.cpp/src/llama-context.cpp#L715-L856)).
`graph_reserve()` rounds token count up to a multiple of sequence count,
builds a dummy ubatch and either splits the graph or asks the scheduler to
reserve its buffers ([same file](../../third_party/llama.cpp/src/llama-context.cpp#L2621-L2683)).
Reservation is therefore a real graph/backend sizing path, not just a small
metadata check.

Before those worst-case graphs, `resolve_fused_ops()` also builds probe
graphs. Its chunked GDN probe is 16 tokens per sequence; the other listed
probes are one token per sequence
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L112-L165),
[probe construction](../../third_party/llama.cpp/src/llama-context.cpp#L599-L674)).
Any probe graph built while paging is required must use the same fail-closed
paged path. A probe larger than the configured cache bound must be reduced to
the supported one-token shape or disabled for that paged context; it must not
silently construct an uncached full-expert graph.

### Graph construction and cache selection

`llm_graph_context` uses all experts as the selection width when
`cparams.warmup` is true, otherwise the model's configured expert count
([`llama-graph.cpp`](../../third_party/llama.cpp/src/llama-graph.cpp#L1535-L1542)).
`build_moe_ffn()` constructs router logits, selects top-k IDs and then asks
`build_moe_cache_slots()` for a slot map
([same file](../../third_party/llama.cpp/src/llama-graph.cpp#L2147-L2253),
[cache-slot builder](../../third_party/llama.cpp/src/llama-graph.cpp#L2281-L2287)).
For supported layers, that helper makes `GET_ROWS(slot_map, selected_ids)` and
uses the returned slot IDs with the cache-bank tensor in `MUL_MAT_ID`
([same file](../../third_party/llama.cpp/src/llama-graph.cpp#L2492-L2531)).

The fallback is implicit: if there is no cache, the slot map is unavailable,
any expert tensor lacks a cache binding, or the target backend rejects the
slot lookup, `build_moe_cache_slots()` returns null. `build_lora_mm_id()` then
uses the original expert tensor and original IDs
([`llama-graph.cpp`](../../third_party/llama.cpp/src/llama-graph.cpp#L1655-L1666)).
For Qwen3-MoE, the model builder reaches this generic routed-FFN path for each
layer ([`qwen3moe.cpp`](../../third_party/llama.cpp/src/models/qwen3moe.cpp#L130-L151)).
Paged mode must turn each of these reasons into context/graph failure. A
missing group or backend op is not permission to use the source tensor.

The existing host MoE cache groups a device's layers only when their complete
expert-tensor lists have matching types, shapes and expert strides
([`llama-moe-cache.cpp`](../../third_party/llama.cpp/src/llama-moe-cache.cpp#L123-L151)).
Its per-group LRU shares slots between layers but keys each resident entry by
layer and expert. The current cache declines a slot-map graph when the ubatch
has more than 32 tokens or its upper bound on distinct selected experts is
larger than that group's slot count
([same file](../../third_party/llama.cpp/src/llama-moe-cache.cpp#L462-L472)).
The caller treats that decline as the generic graph path above.

### Scheduler copy and allocation fallback

The current callback is `bool`. `sched_copy_experts()` first gives the
slot-map/cache handler a chance to handle the source, then has a second
selected-expert copy path
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L2762-L2850)).
At the GGML layer, `false` from the callback means ordinary source copying
continues ([`ggml-backend.cpp`](../../third_party/llama.cpp/ggml/src/ggml-backend.cpp#L1827-L1862)).
That ordinary path copies the full source tensor into a scheduler input copy.
The scheduler reserves graph buffers from graph tensor shapes
([same file](../../third_party/llama.cpp/ggml/src/ggml-backend.cpp#L2062-L2108)),
so copying only the selected byte ranges later does not make the destination
allocation small.

The cache callback currently reads router IDs, synchronizes, plans LRU fills,
uploads missing experts and updates the slot map. It reads expert bytes from
resident `w->data` pointers ([`llama-moe-cache.cpp`](../../third_party/llama.cpp/src/llama-moe-cache.cpp#L515-L582)).
P3/P4 must replace those sources with validated descriptors and bounded reads.
For a source or upload error, P5a's three-way callback result must be
`ERROR`; `NOT_HANDLED` is reserved for ordinary non-expert weights. Returning
false in a required paging context is unsafe.

### Prompt processing, decode and warmup

`llama_decode()` initializes the full logical batch, then calls
`memory->init_batch(..., cparams.n_ubatch, ...)` and processes each returned
ubatch through `process_ubatch()` and `mctx->next()`
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L1879-L1926),
[ubatch loop](../../third_party/llama.cpp/src/llama-context.cpp#L1981-L2003)).
The KV memory module builds the ubatch list before execution
([`llama-kv-cache.cpp`](../../third_party/llama.cpp/src/llama-kv-cache.cpp#L724-L757));
with one stream, `split_simple()` repeatedly takes the first unused token
indices, stopping at `n_ubatch`
([`llama-batch.cpp`](../../third_party/llama.cpp/src/llama-batch.cpp#L585-L617)).
Thus `n_ubatch=1`, one sequence and the normal one-stream causal path preserve
the caller's token order rather than reordering prompt tokens for cache hits.
Each successful ubatch appends its output/token offsets using
`n_outputs_prev`/`n_tokens_prev`; the function advances only after the current
ubatch has completed ([same file](../../third_party/llama.cpp/src/llama-context.cpp#L2124-L2165)).
With one active sequence, the memory module can split a logical prompt into
one-token ubatches in the submitted token order. A large logical `n_batch`
therefore remains usable while physical `n_ubatch` is one.

If graph computation fails, decode removes memory entries starting at the
first failed position for each affected sequence and maps
`GGML_STATUS_ABORTED` to `2`, allocation failure to `-2`, and other failure to
`-3` ([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L2002-L2032)).
Earlier successful ubatches remain in the context. The caller must either
continue from that valid prefix or clear/replay the request before treating it
as an unrelated prompt. A storage error must not produce logits from resident
fallback weights.

`llama_encode()` is different: it calls `split_simple()` once for the whole
input and asserts that `n_ubatch >= n_tokens`; non-causal attention also
requires the whole input in one ubatch
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L1582-L1612),
[decode constraints](../../third_party/llama.cpp/src/llama-context.cpp#L1843-L1898)).
The first paged mode is for causal decoder models only. It must reject encoder,
non-causal and context configurations that would route paged inference through
`encode()`.

There are two warmup paths to account for. Common's `params.warmup` submits
dummy encode/decode batches after context construction and clears/synchronizes
memory afterwards ([`common.cpp`](../../third_party/llama.cpp/common/common.cpp#L1492-L1536)).
It can use one-token decoder ubatches if the paged context fixes
`n_ubatch=1`. Separately, the deprecated `llama_set_warmup()` toggles
`cparams.warmup` without requesting a new scheduler reserve
([`llama-context.cpp`](../../third_party/llama.cpp/src/llama-context.cpp#L1378-L1389)).
That flag currently changes the expert-selection width to all experts. Paged
mode must keep graph selection width at the model's actual per-layer top-k
even during warmup, or explicitly reject this legacy flag. Use real bounded
top-k routing for warmup; if warming every expert is desired later, touch them
one at a time as a separate bounded maintenance operation.

## Minimal correctness configuration

Add an explicit, opt-in `paged_moe_required` mode. Keep it distinct from the
existing `moe_cache_size`/`--moe-cache-mib` host-cache switch: the latter alone
does not mean expert payloads are file-backed or that fallback is forbidden.
For an initial CLI/common path, add `--paged-moe` and require a positive
`--moe-cache-mib`. Keep `-b/--batch-size` as the logical maximum prompt batch;
force or validate `-ub/--ubatch-size 1` and `-np/--parallel 1`. In the library,
set `llama_context_params.n_ubatch=1` and `n_seq_max=1` before construction.
Reject incompatible explicit overrides with an actionable error instead of
silently running a larger physical batch. The PS5LM inference UI does not yet
call `llama_context_params`; it will need a later, separate P6 wiring change.

`n_seq_max=1` is needed in addition to `n_ubatch=1`: the reserve TG graph is
constructed with `n_tokens=n_seqs=n_seq_max`, even though a later runtime
memory context could split a multi-sequence request into single-token
ubatches. One sequence makes both reserve graphs one token and avoids a
reserve-only cache decline. Multiple requests are serialized through one
sequence/context for this first correctness path.

Disable pipeline parallelism for this mode. The existing context already
disables it for `moe_cache_size > 0`, but the new required mode must state the
invariant directly. With one scheduler copy, the current scheduler synchronizes
the split backend before copying each host-weight input
([`ggml-backend.cpp`](../../third_party/llama.cpp/ggml/src/ggml-backend.cpp#L1838-L1849)).
P5a/P5b should make the pager's split-token lifecycle explicit: complete a
previous layer's committed use only after this synchronization, and complete
the final layer token after the graph/backend is synchronized. Do not infer
completion from a CPU cancellation request or from callback return alone.

Before model context construction succeeds, required mode must verify that
every routed expert tensor is supported by the P3 descriptor layout and has a
matching P4 cache bank. Graph reservation, fused-op probes, warmup, prompt
decode and generated-token decode all use the same strict path. If any graph
attempts the original expert source tensor, fails to resolve a slot map, or
cannot upload a selected expert, fail that operation; never fall through to
`build_lora_mm_id()` with source weights or the generic scheduler copy.

## Cache minimum and mixed quant groups

The current group signature is not just one quant type. It is the target
backend/device plus the ordered list of every expert tensor for a layer, with
each tensor's exact type, shape and `nb[2]` stride. For Qwen3-MoE that list is
gate, up and down; other supported graph layouts may have merged gate-up or
extra expert tensors. Layers with a Q4_K/Q6_K difference in any member form a
different group and need their own bank set and slot LRU. A single group has
one shared slot count for all of its weight banks, so the bytes for one slot
are the sum of all member tensor expert strides, not the stride of one matrix.

For one-token ubatches, the conservative lower bound for group `g` is:

```text
slots_min(g) = max over layers l in g of min(n_expert_used(l), n_experts(l))
```

Use the actual per-layer routing width, not a global assumed top-k. When a
future bounded ubatch has `U` tokens, the safe worst-case bound is
`max_l min(U * n_expert_used(l), n_experts(l))`: different tokens can choose
disjoint experts. Duplicate IDs may reduce actual misses but cannot reduce the
capacity guarantee. The initial `U=1` path reduces the minimum to the largest
per-layer top-k in that group's layers.

For each group, count the backend alignment and extra end slot currently
allocated by `llama_moe_cache::alloc_size()`:

```text
minimum_bytes(g) = sum over expert tensors t in g of
                   PAD(t.nb[2] * (slots_min(g) + 1), backend_alignment)
```

Also budget the source staging buffer(s), slot maps, descriptors and fixed
metadata. Do not split the cache budget only in proportion to host expert
bytes and then set an under-minimum group to zero, as the current cache does:
that creates an uncached layer and triggers fallback. First reserve every
group's `minimum_bytes`; fail context creation if their sum plus staging and
metadata exceeds the explicit budget. Only distribute remaining cache bytes
after every group can serve at least one token. Use actual `ggml_type_size` /
`nb[2]` values from the opened artifact, especially for mixed Q4_K/Q6_K
groups; do not estimate all groups from one average expert size.

This per-group minimum assumes P5b's serial split schedule. The cache key
remains `(layer, expert)`. A layer's selected experts are reserved and all
associated tensor banks filled as one transaction; its slot map is published
only after every read and upload succeeds. The previous layer's token can be
released at the next layer callback only after scheduler synchronization proves
the previous split is done. The final layer's token is released after an
explicit backend synchronization. If the eventual adapter cannot provide that
per-split completion proof, it must instead keep all layers' selected IDs
pinned for the whole graph and provision the more conservative sum of their
per-layer bounds per group; it must not assume a slot is safe to reuse merely
because model layers are sequential in source code.

## Errors and recovery

The scheduler copy callback currently cannot distinguish "not ours" from a
file/read/upload error. Reuse the P5a tri-state contract:

```text
NOT_HANDLED  ordinary non-expert input only; generic scheduler copy is allowed
HANDLED      the paged callback filled the cache and slot map
ERROR        a required expert operation failed; stop before executing the split
```

`ERROR` must become `GGML_STATUS_FAILED` (or an equally non-fallback status)
from `ggml_backend_sched_graph_compute_async()`. `process_ubatch()` already
returns a status to decode. Decode then removes KV/state entries for the failed
and later positions of the affected sequence and returns `-3`; retain a
context-owned structured paging error for the app/log so a missing/corrupt
shard is distinguishable from generic compute failure. Graph reservation
failures happen while constructing the context and must make
`llama_init_from_model()` return null. At no stage should failure trigger
`sched_copy_experts()`'s ordinary full tensor copy.

On cancellation between ubatches, preserve only completed prefix positions as
the API currently does; the caller either resumes at the next position or
clears and replays. On a read/upload failure within an ubatch, roll back the
cache reservation, do not publish the slot map, and return failure before the
MoE split executes. If an upload was queued asynchronously, wait for its
completion/fence before rollback can make that storage reusable.

## Focused acceptance tests

Use a synthetic Qwen3-style MoE with deterministic router logits, a tiny
file-backed expert fixture and two layout groups (one Q4_K signature and one
Q6_K signature). Inject read, upload and completion failures. Compare the same
model and exact quantized bytes against a resident reference.

1. **Construction and graph reservation:** create required-paging context with
   `n_ubatch=1`, `n_seq_max=1`. Instrument scheduler/backend allocation sizes
   and source reads through context construction, fused-op probes and all three
   `sched_reserve()` graphs. Assert no allocation or copy is the size of any
   full expert tensor, no expert payload read occurs during graph reservation,
   every routed layer has a slot-bank binding, and all groups meet the computed
   minimum. Repeat with budget exactly equal to the sum of group minima, then
   one byte below; the latter must fail context creation with no uncached group.
2. **Warmup:** exercise common dummy-token warmup and the legacy warmup setter.
   Assert every graph keeps the actual model top-k width and one-token shape,
   both router and cache slot-map paths use the cache banks, and the generic
   full-source copy callback is never called. If legacy warmup cannot be
   constrained, assert it is rejected before graph construction.
3. **Prefill:** submit causal prompts of lengths 1, 2, 17, 33 and the configured
   logical `n_batch` limit with `n_ubatch=1`. Observe exactly one token per
   ubatch and monotonically increasing position/output indices; compare each
   available output/logit with resident inference. Assert total expert source
   bytes equal selected misses only, cache allocations remain fixed, and no
   full expert-tensor allocation/copy occurs. The 33-token case crosses the
   cache's 32-token logical boundary while asserting that every actual cache
   graph still has exactly one token.
4. **Decode and repeat:** decode one token at a time with repeated IDs and
   forced eviction. Compare resident/paged logits, verify a hit avoids reads,
   verify an evicted key is reread, and verify all cache slots remain protected
   until the fake GPU completion token signals. Run a second prompt after a
   completed request to catch stale slot-map or graph-reuse state.
5. **Failure/cancellation:** fail reads, uploads and fences at a chosen prompt
   token. Assert no full-expert fallback, no partially published identity,
   failed-position KV rollback, and valid prefix/replay behavior. Then issue a
   successful repeated request and check there is no stale slot, pin or
   transaction. Include graph reserve and warmup in the no-fallback counter.

The decisive allocation assertion is on backend buffer requests as well as
bytes copied: a callback that copies selected slices into a full-size scheduler
copy still fails. The test should fail if it observes an expert source tensor
as a scheduler destination, if a non-cache MoE `MUL_MAT_ID` is built in
required mode, or if an uncached group reaches decode.

## Tradeoffs and later widening

One-token ubatches keep router IDs and all selected experts exact while
minimizing the per-layer working set. They preserve a larger logical
`n_batch`, but prefill becomes serial: every prompt token traverses every MoE
layer, synchronizes router-ID readback, performs bounded cold reads and waits
for required upload/use completion before the shared group slots can be
recycled. This is intentionally a correctness baseline and may make long
prompts too slow for normal use.

Do not increase `n_ubatch` by guesswork. First measure read, router readback,
upload and compute time. For a chosen `U`, prove that every type/layout group
has `max_l min(U*k_l,n_experts_l)` slots after alignment and staging are
budgeted; reserve and runtime graph shapes must both use that same bound. Add
multi-token parity and completion tests before allowing it. Supporting multiple
sequences also raises the TG reserve and can change the per-layer distinct-ID
bound, so it is a separate extension from larger single-sequence prefill.

The P5b exit criterion is bounded correctness for construction, warmup,
multi-ubatch prefill, decode, cancellation and reuse with zero full-expert
buffer allocations or copies. Console prefill latency and useful product
performance remain measured gates, not consequences of the one-token design.
