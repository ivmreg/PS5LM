# Next stage: run models beyond the resident-memory limit

2026-11-01. Proposed implementation plan; no implementation started.
This replaces the priority order in `next-stage-measured-improvements.md` after
the user clarified that the next stage must enable a better model. Diagnostics
are supporting checks, not the main deliverable. Codex owns architecture,
integration and independent validation. Each numbered worker assignment is for
one Luna agent at maximum reasoning, without nested delegation.

## Concrete objective

Run one complete MoE model whose selected quantization exceeds the console's
resident-memory budget, by reading selected experts from storage into a bounded
GPU cache. Keep the model's original routing, quantized weights and all selected
experts. Compare its useful answers and responsiveness against the working app.

There are two separate success gates:

1. Capability: a previously non-fitting quantized model runs correctly within
   the measured title budget, including load and prompt processing.
2. Product: that model provides a useful quality/latency tradeoff relative to
   the user's current model. More parameters alone do not establish this.

An initial engineering target is **Qwen3-30B-A3B Q4_K_M**: official GGUFs exist,
and the pinned `src/models/qwen3moe.cpp` implements its routed FFN. This is a
tractable port target, not a claim that it beats the user's existing model.
Before acquiring weights, pin the exact artifact, inspect its metadata/tensor
sizes, confirm that the selected quant exceeds the actual budget, and calculate
its non-expert footprint and bytes per selected expert. Use a tiny synthetic
MoE first, so early correctness work requires no large model download.

The subsequent target is the `qwen4exp` model family behind the linked
Betelgeuse experiment, after checking its complete dense/expert/PLE footprint.
The pinned graph already includes routed FFNs and per-layer embeddings, but
their storage and loading must also become bounded. Do not claim the large
bundle fits merely because its stripped dense GGUF is small.

Sources checked 2026-11-01:

- [Official Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B)
  and [official GGUF distribution](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF).
- [Betelgeuse](https://github.com/Betelgeuse-AI/Betelgeuse) describes experimental
  CUDA/NVMe paging and a required PLE companion. Its published PC timings do
  not establish PS5 performance.
- [llama.cpp out-of-core proposal](https://github.com/ggml-org/llama.cpp/pull/31469)
  distinguishes its disk-backed path from the existing host-expert cache.

## What the repository already provides

The pinned llama.cpp has `src/llama-moe-cache.{h,cpp}`, `--cpu-moe`,
`--moe-cache-mib`, graph slot mapping in `llama-graph.cpp`, and scheduler-copy
hooks in `llama-context.cpp`. The cache assumes host-accessible weight tensors;
it is not by itself a bounded disk reader. It also has batch-dependent paths
(`max_batch = 32`) that must be audited for fallback allocations.

Start by extending this mechanism if its invariants permit it. Do not blindly
port the CUDA fork or design a second scheduler. Vulkan implements routed
`MUL_MAT_ID`, but the complete cache path still needs host and PS5 execution
validation. A working ordinary resident MoE does not prove paging correctness.

The PS5 CPU and GPU share physical memory. CPU offload does not create another
large RAM pool. mmap may provide file-backed access but does not guarantee
bounded residency, controlled cache duplication or useful random-read latency.
The implementation must account for those effects instead of promising zero-copy.

## Bounded implementation assignments

### P0. Trace the existing cache and lock the design

Read-only code work; write `docs/design/moe-paging.md`. Inspect model loading,
host weight allocation/mapping, graph reservation, router-ID readback, slot
mapping, expert copies and Vulkan completion. Include decode AND prefill paths.
Pin source references and identify all paths that might allocate or copy the
full expert tensor, including warmup, graph reservation and large batches.

Deliver: exact extension interfaces, supported initial tensor layout/quant,
ownership rules, fallback policy and a budget worksheet for the initial model.
Decide whether the existing cache can accept a file-backed expert source or
needs a narrowly scoped adapter. Codex validates the trace and freezes interfaces
before implementation workers receive write access to upstream code.

Acceptance: every route from load to expert dispatch has an explicit memory and
lifetime explanation. No claim that merely passing cache flags solves capacity.

### P1. Measure the storage path needed by that design

Scope: isolated native-title probe source/build target and protocol notes; no
normal inference changes. Can run alongside P0. Derive representative read sizes
from the proposed expert layout, not only large sequential reads. Read an
explicitly selected existing file, with fixed memory/byte/time/concurrency caps,
seeded offsets and cancellation. Record latency distribution and logical bytes;
label cache state unknown unless actually established.

Codex runs the probe on the target console and uses its raw data to estimate
miss cost at several cache hit rates. Include dense/KV/state/staging/cache in one
shared-memory budget. Missing hardware data does not prevent P2 host work, but
it prevents a claim of usable console speed. Do not benchmark a payload and
present it as a native-title result.

### P2. Implement the expert source with bounded reads

Scope: new isolated expert-source implementation/tests at paths fixed by P0;
no scheduler or Vulkan edits. Prefer indexed reads from ordinary GGUF shards
for the first supported layout when feasible. If packing is necessary, include
a separate offline packer and manifest with exact source identity.

Preserve quantized bytes; validate tensor/expert offsets, alignment, lengths,
shard identity and checked arithmetic. Start with synchronous read completion
and fixed staging buffers. No full-model vector, unbounded mmap fallback,
speculative reads or CPU victim cache. Missing/corrupt bytes fail the request.

Acceptance: byte-identical resident/reader comparisons, short-read/overflow/
truncation/cancellation tests, and a hard measured buffer cap. Codex independently
checks index calculations and failure propagation.

### P3. Make model loading genuinely non-resident for experts

Scope: narrow patch to loader/model tensor setup plus loader tests, fixed after
P0. Dependency: P2 interface. Preserve dense tensors and metadata, but prevent
expert payloads being eagerly allocated/read/uploaded. Retain a validated source
descriptor for every supported expert tensor. Unsupported layouts fail early.

Acceptance: a synthetic expert file larger than the configured budget completes
metadata/load preparation without a proportional host/GPU allocation or full-file
read. Load failure releases resources. Ordinary resident loading stays unchanged.
Do not relax the app's stripped-bundle rejection in this package.

### P4. Connect expert reads to the existing cache

Scope: narrow `llama-moe-cache` patch and focused tests. Dependencies: P0-P3.
Replace the relevant host-memory source access with P2 reads while preserving
the accepted slot interface. Reserve all required experts for a dispatch, retain
their slots through GPU completion, and publish valid slots only after complete
transfers. Define behavior when the selected set exceeds capacity: bounded
chunking with preserved math or explicit rejection, never silent expert loss.

Acceptance: fake completion tests for hits/misses, repeated selection, pressure,
eviction, cancellation and failed reads/uploads. Allocation stays within the
budget; no live slot can be overwritten. Codex reviews the cache state machine.

### P5a. Validate Vulkan transfer and routed single-token execution

Scope: Vulkan adapter/scheduler hook identified by P0 and a small test harness.
Dependency: P4. Use conservative explicit synchronization first. Compare a tiny
resident and paged MoE on the same Vulkan backend at the same quantization,
forcing frequent misses and eviction. Establish tolerances before accepting
results; compare tensors/logits, not only plausible text.

Acceptance: parity under the declared tolerance, no stale slots or buffer lifetime
errors, bounded allocations and native cross-build. Codex repeats execution on
PS5; desktop Vulkan success alone does not pass the PS5 gate.

### P5b. Make prefill and repeated requests bounded

Separate assignment after P5a. Scope: the accepted graph/scheduler path and
focused tests. Exercise graph reservation, warmup, prompt batches and request
reuse. Prevent existing large-batch fallbacks from copying the full expert set.
Begin with explicitly bounded microbatches and one request slot. Support partial
batches, cancellation and long prompts without making asynchronous overlap part
of the first implementation.

Acceptance: resident/paged parity across several prompt lengths; no memory spike
proportional to total experts; valid repeated requests after cancellation. Codex
checks both first load and first real prompt on console.

### P6. Expose an opt-in paged model in PS5LM

Scope: planner/settings and tests first, then a separate dashboard/main wiring
assignment. Dependencies: P3-P5. Preserve default resident behavior. Store an
explicit execution mode/cache budget; plan dense weights, cache, staging, KV,
state and reserve independently from model disk size. Validate the entire model
source before enabling launch. Show experimental status and an actionable error.

Acceptance: no accidental paged launch for ordinary models, settings round-trip,
correct launch arguments, cache-budget refusal and missing-shard rejection.
Codex reviews the actual app call path and rendered UI. Model installation can
initially use FTP; a market-download redesign is not a prerequisite.

### P7. Run the first beyond-RAM model and decide usefulness

One worker prepares the fixed run manifest and result checks; Codex owns console
execution and final interpretation. Use the exact pinned model/quant from P0,
initially at a modest context such as 4k, with all experts preserved. Compare
against its same-quant resident reference on a host with sufficient memory,
and separately compare task quality with the user's existing PS5 model.

Keep the existing benchmark client; add only necessary counters: expert hits,
misses, bytes logically read, read/upload wait, actual cache/staging allocations,
load/prefill/decode and failures. First, repeated and unrelated prompts must be
separate. Reuse a compact quality fixture rather than building a diagnostics
platform before this test.

Proposed initial usability target: sustained decode at least 5 tokens/s on
unrelated requests with reported prefill/first-answer latency. This is a project
target, not a forecast. Report actual results even if it misses. Pass repeated
load/cancellation and a 30-minute mixed-request stability run before deployment.

Capability success requires verified bounded loading and inference beyond RAM.
Calling the model better additionally requires demonstrated useful-answer gains
over the existing model at an acceptable latency cost. If it loses, keep the
engine capability and evaluate a different supported MoE rather than advertise
parameter count as a quality improvement.

## Larger target after the first working pager

P8a: add bounded PLE row reads with resident-byte reference tests.
P8b: integrate PLE and the selected `qwen4exp` bundle, validating every companion
and non-expert allocation; compare resident/paged graph results.
P8c: repeat P7 on the larger model. These are three separate assignments, with
Codex review between them. Do not assume a successful ordinary MoE pager solves
PLE or the dense portion of the larger model.

Optimize prefetch, slot distribution and overlap only after miss/compute traces
identify the limiting cost. Higher weight precision for a supported MoE is also
a valid outcome if it improves answers, without reaching the largest target.

## Execution order and boundaries

P0 and P1 preparation can run in parallel. P2 follows the accepted P0 interface;
P3, P4, P5a, P5b and P6 run sequentially because they touch coupled loading and
execution invariants. Each is a bounded Luna-max assignment with explicit file
allowlists and no nested workers. Codex owns cross-cutting decisions, patch
integration, full host/native checks and package sign-off.

Use the repo's patch workflow; do not silently update llama.cpp or driver pins.
No model download, deployment or benchmark is executed by writing this plan.
The earlier measurement/UI backlog stays available, but cannot delay this
capability work except where a particular correctness or memory check is needed.
