# PS5LM improvement plan

Owner and final reviewer: Codex. Implementation workers: `gpt-6-luna`, maximum
reasoning. Each work package below is one bounded worker assignment; workers
must not spawn other agents. Host checks do not certify console performance.
Existing review: [Betelgeuse feasibility](../BETELGEUSE_REVIEW.md).

## Execution rules

- Preserve the pinned dependencies and pre-existing changes. No dependency
  upgrades, commits, pushes, releases, console deployment or model downloads.
- Assign explicit file ownership. Run independent packages concurrently only
  when their write paths do not overlap; dependent packages run sequentially.
- Each worker returns files changed, design decisions, commands actually run,
  limitations and unresolved questions. Codex inspects actual diffs and runs
  acceptance checks independently before marking a package complete.
- Use synthetic GGUF fixtures and mock HTTP servers for host validation. Keep
  generated output under ignored `build/`; do not check model weights in.
- Hardware-dependent stages require measurements from a console. Never invent
  Pro memory budgets or infer a speedup from theoretical bandwidth.

## First implementation tranche

### 1. Make local model metadata safe to plan

Scope: `ps5/app/model_plan.hpp`, `model_plan.cpp`, `model_plan_test.cpp`.
Input: existing metadata parser, split-file naming conventions and pinned
llama.cpp loader. Keep existing public calls compatible.

Implement explicit split metadata, sum and validate the required local shards,
and refuse incomplete/malformed sets. Reject stripped expert bundles until a
supported pager exists. A remote header alone must not certify a multi-file
model as fitting. Reject invalid metadata before building launch arguments.
Correct failed-fit totals so their advertised context/cache are included.
Preserve resident-model launch defaults.

Acceptance: synthetic single-file success; complete split sizes aggregated;
missing/invalid shards rejected; stripped formats rejected; malformed/truncated
headers rejected; no-fit has no launch arguments and accurate minimum memory;
existing planner configurations still pass. Worker returns parser/API notes
for package 3. Codex compiles and runs the host tests independently.

### 2. Add a reproducible HTTP benchmark client

Scope: new `scripts/benchmark-ps5lm.py`, `tests/test_benchmark_ps5lm.py`,
`docs/BENCHMARKING.md`. Independent of package 1.

Implement a Python standard-library client against the already-running
server's streaming OpenAI API. Require an explicit URL. Run fixed prompts
with deterministic sampling, an optional separately reported warmup, repeat
and unrelated-prompt trials. Persist machine-readable results, model/server
identity where available, request parameters, first-content latency, wall time,
server timing/token counts when provided, and errors. Distinguish client
measurements from server decode rates. Do not fabricate memory, physical I/O,
load time or token counts when the API does not supply them. Bound requests
with a timeout and exit unsuccessfully on failed trials.

Acceptance: mock server covers fragmented SSE, DONE, errors, missing usage,
timeouts and warmup separation; CLI help works; results retain failed trials.
Documentation states what needs separate hardware instrumentation. Codex
reviews formulas and independently runs the mock-server suite.

### 3. Make remote model checks conservative

Depends on package 1. Scope: `ps5/app/market.cpp`, `market.hpp`, dedicated host
tests, and the dashboard's unsupported-market badge/action handling only.
Keep single-file search/download behavior unchanged.

Use package 1's explicit metadata status for multi-file/stripped models. Until
the downloader supports a validated complete manifest, mark split downloads
unsupported instead of offering the first shard as a complete model. Show an
actionable explanation. Do not mark unsupported cases as transient header
failures or enable their download action. Validate the real dashboard/job
call path, not just a new helper.

Acceptance: single-file flow preserved; split/stripped models cannot receive a
positive fit or a single-file download job; status remains understandable;
syntax/host checks cover changed code. Codex reviews the integration.

### 4. Expose transparent planning profiles

Depends on package 1 and integrates after package 3. Scope: planner, settings,
dashboard and main call sites, with focused planner/settings tests.

Execute as two Luna chunks: 4a owns planner policy, settings persistence and
their tests; 4b follows review of 4a and owns only dashboard/main wiring and
focused UI integration checks. Neither writer overlaps the market worker's
dashboard changes.

Keep the existing longest-context policy as the default for old settings.
Add balanced and higher-KV-precision policies with explicit, documented
candidate ordering and target context. Honour the user context cap and forced
KV type in every policy. Label the precision choice as KV cache precision,
not a promise of higher model-weight precision or better answers. Profile
selection must persist and update local plans without changing weights.

Acceptance: default backward compatibility; different policies select expected
configurations for a constrained synthetic budget; forced cache/caps obeyed;
unknown profile falls back safely; settings round trip; UI/main wiring builds.
Codex reviews policy tradeoffs and runs the tests independently.

### 4c. Keep the host checks in CI

Independent integration chunk after test interfaces are fixed. Scope:
`scripts/test-host.sh`, `.github/workflows/native-build.yml`, `docs/CI.md`.
Build/run the planner, mocked market and settings checks; syntax-check the
dashboard; run Python build-queue and streaming-benchmark suites. Initialize
only pinned source submodules in CI, keep local execution network-free, and
preserve the separate full native cross-build job. Acceptance: Codex executes
the host runner and reviews workflow changes; no remote CI result is implied.

## Measurement-gated follow-on packages

### 5. Capture console memory and storage baselines

Depends on packages 1–4. Worker scope: one new diagnostic script/probe and its
documentation, with explicit PS5 APIs selected from existing probes.
Measure native-title free memory before/after loading, allocation peaks where
observable, sequential and random reads with bounded sizes/concurrency, and
the same model/prompt benchmark on available hardware. Keep payload and title
results separate. No inferred extra Pro memory.

Acceptance: bounded runs, machine-readable raw observations, hardware/firmware
and storage identity, unsupported counters marked unavailable. Codex validates
host behavior; console acceptance remains pending until actually executed.

### 6. Revise memory estimates from measured evidence

Depends on package 5. Worker scope: planner reserve policy and regression
fixtures from the baseline. Separate weights, KV, recurrent state, checkpoint,
vision, compute and safety allowance. Preserve an explanation for each term.
Do not replace estimates with unmeasured per-console constants.

Acceptance: calculations reproduce observed configurations with documented
headroom; no regressions on current supported models; uncertain architectures
remain explicitly estimated. Codex independently checks arithmetic and data.

### 7. Define and test an expert-store reader

Depends on an accepted package 5 go/no-go decision. Worker scope: isolated,
backend-independent reader plus a tiny synthetic pack and tests; no scheduler
changes. Specify quantized payload preservation, validated offsets/alignment,
bounded buffers, short-read/corruption handling and cancellation.

Acceptance: exact byte round trips and deterministic failures under truncation,
bad indices and cancellation; no arbitrary model-format acceptance added.
Codex reviews format and lifetime guarantees before integration.

### 8. Prototype a synchronized Vulkan expert cache

Depends on package 7. Execute these as separate Luna assignments, with Codex
review between them. Before each writer starts, Codex fixes the exact file
allowlist and interface using the preceding discovery result.

- **8a, read-only integration map:** identify the routed-weight access,
  scheduler synchronization and Vulkan buffer-lifetime call sites in the
  pinned backend. Return source references and one proposed minimal hook;
  do not implement a second scheduler. Acceptance: Codex can trace an expert
  selection through dispatch and knows where data must become visible.
- **8b, bounded cache policy:** implement a backend-independent slot allocator
  around package 7's reader, with fake GPU completion fences. Acceptance:
  deterministic tests prove live slots cannot be evicted, all requested
  experts arrive, and allocation stays inside the configured byte budget.
- **8c, Vulkan slot transfer:** implement only allocation, upload and explicit
  completion for those slots. Acceptance: a small synthetic tensor survives
  upload/reuse and CPU-reference comparison on an available Vulkan host;
  failed transfers never publish valid slots. PS5 checks remain a separate gate.
- **8d, routed-MoE integration:** wire the accepted slot API into one selected
  small MoE's ordinary synchronized execution. Acceptance: paging-on/off on
  the same backend and quant matches within a declared numerical tolerance;
  misses, evictions, cancellation and repeated requests are covered.

No speculative prefetch, doorbells, Linux-only I/O or performance promises in
these chunks. Codex owns the pinned-source build and console go/no-go review.

### 9. Handle PLE and evaluate a larger model

Depends on package 8 correctness. Execute as three separate Luna assignments:

- **9a, PLE reader:** add bounded row reads/staging with synthetic n-gram/row
  fixtures; no graph changes. Acceptance: resident-reference bytes match,
  out-of-range rows fail, and buffers never grow past the declared cap.
- **9b, companion and graph integration:** validate the complete bundle and
  wire the accepted PLE staging API into the selected model graph. Acceptance:
  missing companions fail before load; repeated-row and uncached-row paths
  agree; all resident/paged allocations enter the memory plan.
- **9c, controlled console evaluation:** run the benchmark protocol at matched
  quants/contexts on first, repeated and unrelated requests. Return raw memory,
  I/O and timing observations and same-backend correctness comparisons.
  Acceptance: no OOM or stale-slot output, bounded memory and useful measured
  sustained speed. Codex checks raw records and decides deployability.

Any overlap optimization is a new bounded package justified by the measured
bottleneck. Model size alone is not success.

## Status

This section records the original implementation handoff. Subsequent local
build manifests record a completed native build and package, and the user has
confirmed that the solution works after completing it with another agent.
The [next-stage plan](next-stage-measured-improvements.md) starts from that
working baseline; the validation limitations below describe the earlier handoff.

- Packages 1–4 and CI integration are implemented in the working tree by Luna
  workers at maximum reasoning, with Codex integration/review. Existing
  `long_context` behavior remains the default; profiles do not change weights.
- Codex independently ran `bash scripts/test-host.sh` under WSL: planner,
  mocked market, settings, dashboard syntax, seven build-queue/preflight tests
  and thirteen benchmark tests all passed. The benchmark's thirteen tests
  also passed on native Windows, plus an independent chunked-transfer probe.
- Codex ran the final planner suite with undefined-behavior sanitization;
  it passed. A separate sanitized market integration run also passed.
- Shell source syntax and Python compilation passed. The existing Windows
  checkout uses CRLF for tracked shell files, so shell source was checked with
  CRLF normalized in the validation stream; Git settings/files were not
  rewritten. The new host runner executes directly through Bash.
- No trained model, full native cross-build, rendered UI preview, console
  inference or performance comparison was run. Native/rendered/hardware
  validation remains required before treating this tranche as deployable.
- Packages 5–9 remain measurement-gated follow-on work, not promises of a
  large-model speedup. No commit, push, release or console deployment was made.
