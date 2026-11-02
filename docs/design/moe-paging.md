# Bounded MoE expert paging: P0 design

Status: source trace and interface proposal, 2026-11-01. No inference code is changed.

## Scope and pins

The implementation target is the official `Qwen3-30B-A3B-Q4_K_M.gguf`, HF
repository revision `e4d4bafdfb96a411a163846265362aceb0b9c63a`, LFS SHA-256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`, file
size 18,556,685,824 bytes (17.282 GiB). The pinned llama.cpp source is commit
`04f7226a2c12c7f382c545a6db5a034266aa53cb` at `third_party/llama.cpp`.
`third_party/llama.cpp/AGENTS.md` applies to changes inside that submodule.

The target weights are not in this workspace; only llama.cpp vocabulary GGUFs
are present. Therefore the artifact's per-tensor types, dimensions, offsets,
exact expert bytes, and exact non-expert bytes remain to be read from that
specific file before P3. The design below does not treat file size as memory
size or claim that `--cpu-moe` plus `--moe-cache-mib` enables beyond-RAM use.

Update: the [pinned target header audit](qwen3-target.md) has since measured
these tensor dimensions and offsets without downloading payload bytes. Down
experts use Q4_K in 24 layers and Q6_K in 24; cache accounting must use each
tensor's actual stride. Payload integrity is still unverified.

## Findings from the pinned source

### Loader and host allocation

`llama_model_loader::llama_tensor_weight` already records source shard index,
absolute tensor data offset, and tensor metadata (`src/llama-model-loader.h:27-41`).
`create_tensor` validates tensor shape, selects a buffer type, creates a GGML
tensor, and later `llama_model_base::load_tensors` allocates a backend buffer
for each context and calls `load_all_data` (`src/llama-model-loader.cpp:1117-1390`,
`src/llama-model.cpp:1912-2014`). Qwen3-MoE currently creates three separate
expert tensors per layer, all with `flags = 0`: gate/up `[n_embd,n_ff_exp,n_expert]`
and down `[n_ff_exp,n_embd,n_expert]` (`src/models/qwen3moe.cpp:45-54`).
`llama-arch.cpp:978-984` maps those expert tensors to `GGML_OP_MUL_MAT_ID`.

`--cpu-moe` adds a CPU buffer override for all expert weights; `--n-cpu-moe`
can limit it by layer (`common/arg.cpp:2763-2778`). This changes placement, not
allocation: CPU expert tensors still receive full payload buffers and the
loader reads or maps their full tensor ranges. On PS5, CPU and GPU allocations
draw from shared physical memory.

There is an existing `TENSOR_READ_LAZY` path, used by Gemma 4 per-layer token
embeddings and Qwen4Exp PLE (`src/models/gemma4.cpp:59`,
`src/models/qwen4exp.cpp:214`). It selects CPU buffer type and records lazy
ranges (`src/llama-model-loader.cpp:1081-1114,1343-1346`). With mmap buffer
support, lazy tensor data can point into the file mapping and is excluded from
the ordinary mmap buffer ranges; it avoids an eager copy and lock
(`src/llama-model.cpp:1912-1944`, `src/llama-model-loader.cpp:1491-1520,1633-1667`).
It still maps the whole file, relies on OS page faults and reclaim, has no
bounded expert staging buffer or explicit expert-page eviction, and can fall
back to ordinary allocation when mmap support/buffer wrapping is unavailable.
It is useful as a prototype or reference, but cannot pass the bounded-residency
gate by itself.

The loader has `load_data_range(weight, offs, size, buf)`: without mmap it seeks
to the requested range and reads into caller storage; with mmap it returns a
pointer into the mapping (`src/llama-model-loader.h:235-246`,
`src/llama-model-loader.cpp:1522-1547`). This is the closest existing primitive
to the required read, but the loader owns its shard handles and does not survive
model loading. A runtime source must own durable shard readers (or equivalent
platform handles) and validated descriptors.

### Existing expert cache and router callback

`llama_moe_cache` only registers layers whose expert tensors are host weights.
It groups layers with exactly matching tensor type/shape/stride, allocates GPU
banks and host slot maps, and divides the configured budget among groups and
devices (`src/llama-moe-cache.cpp:130-151,221-328,353-455`). Each cached tensor
bank reserves `n_slots + 1` experts for backend padding. The slot LRU can fit a
single decode token only when it can hold at least `n_expert_used`; a selection
with more distinct IDs than slots currently reaches `GGML_ABORT`
(`src/llama-moe-cache.cpp:72-119,315-328,552-555`).

The graph computes router logits and top-k IDs, creates `GET_ROWS(slot_map, ids)`,
and substitutes cached bank tensors into `MUL_MAT_ID` when the cache path is
available (`src/llama-graph.cpp:2147-2253,2509-2552`; Qwen3 calls it at
`src/models/qwen3moe.cpp:125-139`). Scheduler creation registers the copy
callback (`src/llama-context.cpp:702-712`). For a slot-map copy, the cache
callback finds the `GET_ROWS`, reads router IDs with
`ggml_backend_tensor_get_async`, synchronizes the backend, plans LRU fills,
uploads missing expert slices from `w->data + expert * w->nb[2]`, then uploads
the slot map (`src/llama-moe-cache.cpp:515-585`). Thus this callback already
provides the correct dispatch point, but its source is a resident host pointer.

The generic `sched_copy_experts` path also handles uncached/large-batch
`MUL_MAT_ID`: it reads IDs back, synchronizes, and copies selected expert slices
into a scheduler copy with the same full tensor layout (`src/llama-context.cpp:2762-2852`,
`ggml/src/ggml-backend.cpp:1528-1535,1827-1862`). The destination can therefore
reserve a full expert-tensor-sized device allocation even when only selected
experts are copied. A paged mode must never reach this fallback.

### Vulkan synchronization and graph reservation

Vulkan dispatches `GGML_OP_MUL_MAT_ID` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:12724-12730`)
and advertises operation support subject to type and ID-shape constraints
(`:15453-15469`). `ggml_backend_vk_set_tensor_async` can enqueue a host-to-device
write and defer copying host bytes until synchronization; fallback paths use
shared staging and synchronize (`:13220-13310,13414-13455`). Router-ID readback
already synchronizes in the cache. A disk-reader staging buffer must remain
owned and unchanged until its upload is complete. The first implementation
should use explicit synchronous upload completion; asynchronous read/upload
overlap needs a later measured lifetime protocol.

`sched_reserve` reserves a prompt-processing graph using `n_tokens = min(n_ctx,n_ubatch)`,
a token-generation graph using `n_seqs`, then reserves the prompt graph again
(`src/llama-context.cpp:715-856`). Graph reservation allocates scheduler copies;
it is not a harmless metadata-only pass. Cache selection is declined when
`n_tokens > 32` or the upper bound `min(n_tokens*n_expert_used,n_expert)` exceeds
the slot count (`src/llama-moe-cache.cpp:196,462-472`). With common prompt
microbatches above this limit, graph reservation and real prefill can take the
full-shape fallback. `sched_graph_realloc` only counts/logs reallocations; it
does not enforce the pager budget (`src/llama-context.cpp:2712-2730`).

## Smallest feasible interface

Retain the existing GGML tensor shape and router/cache graph. For supported
expert tensors, create GGML metadata without allocating or reading their payload.
Keep a model-owned source descriptor for each tensor:

```text
ExpertTensorDesc {
  tensor_id, shard_id, absolute_data_offset, tensor_bytes,
  ggml_type, ne[4], expert_stride_bytes, source_file_size/identity
}
read_expert(tensor_id, expert_id, dst, dst_capacity) -> ReadStatus
```

`read_expert` must check the expert index, checked offset arithmetic, expected
length/capacity, shard identity and short reads. It copies the exact quantized
bytes for one expert into a fixed-size staging buffer; it never dequantizes,
reads the whole tensor, creates an unbounded mmap fallback, or prefetches. An
internal implementation can reuse the loader's file-offset semantics, but must
own readers after the loader is destroyed. For initial Qwen3-MoE, `expert_stride`
is `tensor->nb[2]`, with no layout conversion. Support only the three gate/up/down
GGUF tensors in this family; reject split-expert, merged gate-up, PLE, and
unsupported layouts until separately implemented.

Pass the source to `llama_moe_cache`; on a cache miss read each selected expert
for each bank tensor into bounded staging, upload, wait for completion, and only
then publish the slot map. Make fill/eviction transactional: a read/upload error
invalidates any newly reserved slots and fails the current request. Keep all
selected slots protected until graph use completes. For P5a, use one request,
one token per microbatch, and synchronous uploads. That preserves all routing
and experts while avoiding the batch fallback; it is a correctness baseline,
not a speed recommendation. Increase prefill microbatch size only when the
worst-case distinct selected IDs fit the cache or a bounded exact chunking
scheme is implemented.

### Scheduler error propagation required for missing/corrupt reads

The current scheduler callback type returns `bool`, where false means “copy the
whole weight normally” (`ggml/include/ggml-backend.h:320-327`; use is at
`ggml/src/ggml-backend.cpp:1847-1862`). That cannot represent a hard read error
without either unsafe fallback or process abort. Add a three-way callback result
(`NOT_HANDLED`, `HANDLED`, `ERROR`) and thread it through
`ggml_backend_sched_copy_input` to `ggml_backend_sched_compute_splits`. On
`ERROR`, stop before executing that split and return `GGML_STATUS_FAILED` from
`ggml_backend_sched_graph_compute_async` (whose status return already exists at
`ggml/include/ggml-backend.h:361-365`, implementation at
`ggml/src/ggml-backend.cpp:2118-2131`). `llama_context` retains the structured
read error for the caller. A failed source read must not retry through resident
weights. Existing callers map to NOT_HANDLED/HANDLED without behavior change.

## Initial budget worksheet

The pinned artifact file size is exact: **18,556,685,824 bytes**. The local repo
has no copy to inspect, so exact dense/non-expert footprint and exact bytes per
expert are still unmeasured. Use GGUF metadata and tensor offsets from the exact
LFS-pinned file before enabling loading.

For expected Qwen3-30B-A3B config (48 layers, hidden 2048, expert FFN 768, 128
experts, top-8; verify all five from the artifact metadata):

| Quantity | Calculation | Value / state |
|---|---|---:|
| Total GGUF file | HF LFS size | 18,556,685,824 B (17.282 GiB) |
| Expert logical parameters | `48 * 128 * (2*2048*768 + 768*2048)` | 28,991,029,248 params |
| Bare 4-bit expert payload floor | expert params / 2 | 14,495,514,624 B (13.50 GiB); actual Q4_K_M bytes are larger due to block scales/mixed tensor types |
| Exact expert bytes | sum GGUF byte lengths for gate/up/down expert tensors | unknown until pinned artifact metadata is read |
| Exact non-expert + GGUF overhead | file bytes - exact expert bytes | unknown; upper bound 4,061,171,200 B if expert tensors were exactly at the bare 4-bit floor |
| Selected logical expert payload per token | `8/128 * exact expert bytes` across all layers | unknown; bare 4-bit floor 905,969,664 B (864 MiB) per cold token |
| One expert's bytes/layer | gate + up + down, each from exact GGUF tensor type/stride | `2*768*row_size(type,2048) + 2048*row_size(type,768)`; verify per tensor |
| Device cache | configured bytes, plus per-bank alignment and extra slot | explicitly measured at context creation |
| Host read staging | fixed buffer count x maximum supported expert bytes | fixed in P2; included in shared-memory budget |
| Remaining runtime | dense/model buffers + compute graph + KV/state + app reserve + cache + staging | must fit measured PS5 title budget; P1 measures file-read latency, console run measures memory |

This floor proves the artifact is plausibly beyond a small console budget, but
does not prove the exact target's expert share or fit. On PS5, do not count CPU
expert pages as a second RAM pool. Count non-expert model memory, GPU cache,
read staging, graph buffers, KV/state, app/title reserve, and any OS file cache
against one shared physical-memory budget. mmap residency remains an unknown,
not a zero-cost line item.

## Staged implementation boundary

1. **P2 source:** add the bounded reader, descriptor validation, fixed staging,
   read failure results and byte-identical/failure tests. No scheduler changes.
2. **P3 loader:** preserve tensor metadata and source descriptors while avoiding
   expert payload allocation/load. Reject unsupported or incomplete sources
   before model launch. Keep default resident loading unchanged.
3. **P4 cache:** consume reader misses in existing slot banks; make reserve/fill/
   publish transactional and retain slots through compute completion. Fail if
   selected IDs exceed slot capacity.
4. **P5a scheduler/Vulkan:** add callback error result propagation and explicit
   upload completion; compare resident and paged tiny MoE logits on Vulkan.
5. **P5b prefill:** lock `n_ubatch`/`n_seq_max` to the supported bound so graph
   reservation and real prefill do not allocate full expert copies. Later add
   bounded microbatch retry or exact chunking. Exercise warmup, cancellation,
   first prompt and repeated requests.

Initial file allowlists:

- P2: new `src/llama-expert-source.{h,cpp}`, source tests and only the build
  registration those files need.
- P3: `src/llama-model-loader.{h,cpp}`, `src/llama-model.{h,cpp}`,
  `src/models/qwen3moe.cpp`, plus focused loader tests.
- P4: `src/llama-moe-cache.{h,cpp}`, `src/llama-context.{h,cpp}`.
- P5a: `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`,
  `src/llama-context.{h,cpp}`, and the Vulkan backend only if target execution
  shows the existing upload path cannot provide the required completion fence.
- P5b: context graph sizing/reservation and focused inference tests only.

Do not relax PS5LM's stripped-bundle rejection, update the llama.cpp or driver
pin, or broaden this initial support to PLE. Capability passes only when load,
graph reservation, prefill and decode have no allocation proportional to all
experts; useful quality and latency are a separate gate.
