# PS5LM review against Betelgeuse

Reviewed 2026-11-01. Repository: `ivmreg/PS5LM`, commit
`0d1b70403086ab762ac0ea7a510ca9acd8a4c770`; pinned llama.cpp:
`04f7226a2c12c7f382c545a6db5a034266aa53cb`. All three submodules were initialized
at their recorded revisions. This is a source and feasibility review, with host
checks; no PS5 inference or Betelgeuse performance was reproduced.

The [linked Reddit post](https://www.reddit.com/r/Qwen_AI/s/3RPoVahmuG)
describes out-of-core MoE inference. Its ideas are relevant to PS5LM's
SSD-streaming roadmap, but adopting the engine requires new PS5/Vulkan work.
The existing app remains a resident-model implementation.

The subsequent [implementation plan](plans/ps5lm-improvements.md#status) records
working-tree fixes for the sizing and unsupported-bundle findings below,
planning profiles, benchmark tooling and Codex's validation. The findings in
this review describe the original commit named above.

## Findings

1. **High: split GGUF memory is undercounted.**
   `ps5/app/model_plan.cpp:202` obtains the selected file's size;
   line 329 uses that size plus an optional image projector as total weights.
   `find_models` at line 415 deliberately lists only the first shard, but the
   metadata parser never aggregates `split.count` companions. The underlying
   llama.cpp loader does open additional shards
   (`third_party/llama.cpp/src/llama-model-loader.cpp:596`). Consequently, the
   app can offer a model that cannot fit its stated budget. A synthetic header
   declaring two shards, with a 6 GiB first shard, was accepted under an
   11.38 GiB budget. A second 6 GiB shard would put weights alone over budget.
   Aggregate and validate all shards before reporting fit; remote model checks
   need the same accounting.

2. **High for importing this bundle: stripped models can receive a false fit.**
   The planner reads architecture/layer metadata but skips the tensor directory
   (`model_plan.cpp:169`) and does not recognize stripped expert metadata or its
   companion files. A synthetic `qwen4exp` header with
   `betelgeuse.stripped_experts=1` and a 3,770,978,720-byte dense file was accepted
   as 3.51 GiB of weights. This demonstrates planner acceptance, not successful
   loading of the published bundle. Reject unsupported stripped bundles with
   a clear message until their loader, dependency validation and pager exist.
   The author's [bundle documentation](https://github.com/Betelgeuse-AI/Betelgeuse/blob/main/README.md#architecture)
   explicitly requires its custom pager and a separate PLE companion.

3. **Implementation blocker: architecture support is already present, streaming is missing.**
   The pinned `src/models/qwen4exp.cpp` has routed FFNs and PLE handling
   (lines 1142 and 1349). Adding a model-name preset will not implement demand
   paging. `scripts/build-app.sh:88` selects Vulkan; the planner starts ordinary
   loading with `-ngl 999 -fit off -lm none` (`model_plan.cpp:385`). No Betelgeuse
   paging arguments or expert-store integration were found in this path.
   The [upstream RFC](https://github.com/ggml-org/llama.cpp/pull/31469)
   adds an expert store, cache policy and GPU/host coordination; integration
   affects the backend and scheduler as well as the app.

## What transfers to the PS5

| Betelgeuse idea | PS5LM implication |
| --- | --- |
| Bounded expert cache; preserve every selected expert | Good starting design. Make misses fetch weights rather than prune routing. |
| Separate host victim cache and VRAM slots | Redesign around shared memory. CPU cache, GPU weights, KV/state and staging consume the same console budget. |
| Packed expert reads | Useful candidate; measure random-read latency, concurrency, alignment and page-cache duplication in the native title. |
| CUDA graph/doorbell coordination | Requires a Vulkan/PS5 implementation with explicit visibility and lifetime guarantees. |
| Linux `io_uring` worker | Requires a PS5-supported I/O implementation; begin with a bounded read worker and measure it. |
| Profile-guided slot distribution | Tune using PS5 traces rather than copying 5,000 slots or the RTX layer allocation. |

The console notes report an 11.44 GiB GPU heap and about 11.03 GiB free after
the dashboard (`docs/CONSOLE.md:116`, line 136). CPU memory shares that pool.
An additional 8 GiB victim cache cannot be treated as separate capacity.
PLE also needs a bounded row-access strategy; streaming experts alone does
not solve the model's complete residency problem. The current PLE graph gathers
rows from `per_layer_tok_embd` (`qwen4exp.cpp:1362`).

Do not assume fewer GPU submissions alone will accelerate PS5LM: the project's
existing experiments did not improve generation (`patches/mesa/README.md:12`).
Benchmark a correctness-first, synchronized pager before optimizing overlap.

## Interpreting the advertised performance

The author's [measurement record](https://github.com/Betelgeuse-AI/Betelgeuse/blob/main/docs/fluxaeros.md#measurements-and-interpretation)
qualifies the headline: 16.80 and 22.03 tok/s come from different historical
runs; the former paid generated startup warmup, and the latter repeated a
workload. The recorded envelope used context 512. Historical token/router
parity does not establish bitwise parity across CUDA and Vulkan, or certify a
PS5 port. Compare paging on/off on the same PS5 backend and quantization.

The current [README](https://github.com/Betelgeuse-AI/Betelgeuse/blob/main/README.md#get-the-four-file-bundle)
also documents roughly 68 GB of runtime files, including the expert store and
PLE; the small dense GGUF is not the complete model. Its optimized reads are
buffered despite the `--moe-direct-io` name. These details matter when designing
memory accounting and measuring physical storage traffic.

## Recommended next work

1. Correct shard accounting and explicitly reject unsupported stripped formats.
2. Measure native-title SSD reads and available shared memory under inference.
3. Prototype a small, opt-in Vulkan expert pager with a bounded unified pool,
   validated pack offsets and fatal handling of missing/corrupt expert bytes.
4. Add PLE row staging, then optimize overlap only after correctness checks.
5. Measure first request, repeated prompts and unrelated prompts separately;
   include warmup cost, prefill, decode, memory peaks and bytes read per token.
   Exercise eviction, short reads, cancellation and repeated requests. Keep
   speculative decoding disabled for the initial comparison.

## Validation performed

- Existing `model_plan_test`: passed, synthetic headers only.
- Existing `tests/test_build_app.py`: all seven tests passed under WSL Ubuntu.
- Two additional synthetic planner probes reproduced findings 1 and 2. Their
  source and executable are in ignored `build/review/`, alongside the existing
  planner-test executable.
- Full PS5 cross-build, real model loading, console I/O measurements and
  throughput/parity benchmarks were not run. No inference implementation was
  changed as part of this review.
