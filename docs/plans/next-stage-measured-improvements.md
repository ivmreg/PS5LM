# Next stage: measured model quality, performance and fit

Priority superseded by [Models beyond RAM](beyond-ram-models.md). The user
clarified that the next stage must enable a better model, not primarily improve
measurement of models that already fit. Retain this document as supporting work;
it is not a prerequisite sequence for the pager implementation.

Planned 2026-11-01. Status: proposed, implementation not started.
Owner, integrator and final reviewer: Codex. Implementation packages are sized
for one `gpt-6-luna` agent at maximum reasoning, with no nested delegation.
This document is a plan, not an instruction to start agents or change the console.

## Outcome and baseline

The user confirms that the implementation and deployment work completed with
another agent and the solution works. Treat that build as the working baseline.
Do not reopen the completed first tranche just because its earlier notes predate
the successful build. Existing host/native checks remain regression gates.

The next deliverable is an evidence-backed answer to: **which model, weight
quantization, context and KV configuration gives this console the best useful
answers at acceptable latency?** Ship better defaults only where the measurements
support them. A larger model, longer context or higher token rate alone is not
the success criterion.

Repository evidence informing this plan:

- `model_plan.cpp` already handles local shards and planning profiles, but its
  reserve combines a 0.75 GiB allowance, a recurrent-state checkpoint, and guessed
  vision/wide-batch allowances. These need attribution before tuning.
- `main.cpp::log_memory` and `stats.cpp::StatsCollector::run` already read direct
  memory and heap information. Reuse these paths rather than creating competing
  samplers. Verify API semantics and return codes before interpreting readings.
- `stats.hpp` describes GPU activity derived from server compute time. This is
  a proxy, not a hardware GPU-utilization measurement.
- `benchmark-ps5lm.py` already captures streaming timings and raw server fields,
  but has fixed short prompts and no answer-quality or memory comparison.
- Model switching deliberately restarts the title to reclaim memory; retain
  that behavior while gathering data. Do not implement in-process reload here.
- Remote split downloads are intentionally blocked; local complete splits work.
- `build/native-deploy/BUILD-INFO.json` records a completed local native build
  with embedded browser assets. Its earlier static-validation JSON describes a
  different, browser-assets-omitted binary. Future results must bind to the exact
  final artifact, not just the app version.

No new model ranking or PS5 Pro capacity claim is made here. Model candidates
will come first from files already available to the user. Record actual hardware;
if only one console is available, evaluate that console without extrapolation.

## Execution and review rules

Each package below is a separate assignment with the listed write scope. A
worker reads related code freely but does not edit other paths. Before dispatch,
Codex resolves any proposed filename against the current tree and freezes public
interfaces needed by dependent tasks. No worker commits, pushes, deploys, changes
dependency pins or downloads weights. Preserve all existing working-tree changes.

Each return includes the actual diff, commands/results, fixtures and limitations.
Codex independently inspects the diff and runs relevant checks. Integration files
such as `scripts/test-host.sh`, CMake/build scripts and the CI workflow have one
writer at a time. Console measurements and final acceptance belong to Codex;
workers can prepare tooling and analyze copied records, not certify hardware
they did not run. Do not mark a hardware gate passed from host mocks.

## Milestone A: make results attributable and comparable

### N0. Record the known-good baseline

Scope: new `docs/validation/working-baseline.md`; read existing build manifests,
checksums and test reports. No code changes. Dependency: none.

Record source commit plus dirty-source digest, submodule revisions, final package
hash, final executable hash, and the user's success confirmation as user-reported
evidence. Keep missing hardware/firmware/model details explicitly unknown. Link
the historical first-tranche plan and distinguish pre-UI static checks from the
final packaged binary. Do not change historical manifests retrospectively.

Acceptance: every technical claim traces to a file or an attributed observation;
another run can identify whether it used this exact baseline. Codex verifies hashes.

### N1. Define a small diagnostic record and bounded writer

Scope: new `ps5/app/diagnostics.hpp`, `diagnostics.cpp`, `diagnostics_test.cpp`,
and `docs/DIAGNOSTICS.md`. Dependency: N0; no platform calls in this package.

Define versioned JSONL records for session identity, configuration, load start,
ready, request start/end, periodic samples and errors. Include a run ID, monotonic
time, build ID, model identity, effective launch arguments and metric provenance.
Represent unavailable readings as null with a reason, never as zero. Distinguish
sampled maxima from allocator high-water marks; neither proves a GPU peak.

Implement an opt-in bounded writer with a maximum of 16 MiB per session and
64 MiB total in a dedicated diagnostics directory. Keep serialization/file I/O
off render and inference callbacks; bound the queue, count dropped records, and
handle full storage without blocking inference. Record no prompt text or answers
in app diagnostics. CPU heap and GPU accounting may overlap in shared memory;
do not sum them as independent physical pools.

Acceptance: synthetic tests cover schema, nulls, escaping, queue/cap limits,
disk-write failure and concurrent producers; shutdown ownership is explicit.
Codex reviews thread lifetime and backpressure before any integration.

### N2. Wire diagnostics into existing lifecycle and samplers

Scope: `ps5/app/main.cpp`, `stats.cpp`, `stats.hpp`, and only the build-list
entries needed for N1. Dependency: accepted N1; one integration worker.

Add session/build/model identity, effective plan and load-start/ready timestamps.
Reuse the existing one-second sampler for raw direct-memory/heap readings,
server counters and available sensors. Check return codes and bounds; retain
raw source identity. Investigate whether the kernel availability value denotes
total free capacity or a largest allocatable region and label it accordingly;
leave interpretation unknown if the source cannot establish it.

Expose collection through an opt-in local marker/control request consistent
with existing app control files. Export records using the existing FTP retrieval
path; do not add a new public HTTP service. Correlate request IDs where existing
hooks permit it, otherwise report only the enclosing run/time interval. Make
readiness mean the server is usable, not just that a socket opened. Carry a run
ID across intentional LoadExec restarts without replaying old control requests.

Acceptance: host fake-source/lifecycle checks; native cross-build; baseline runs
with collection off/on. Proposed overhead budget: at most 2% median latency or
decode-rate change, measured with interleaved repeats. No claim of overhead
compliance when noise or missing measurements prevents a conclusion.

### N3. Extend benchmark workloads and write a strict comparison tool

Scope: `scripts/benchmark-ps5lm.py`, `tests/test_benchmark_ps5lm.py`, new
`scripts/compare-ps5lm-runs.py`, `tests/test_compare_ps5lm_runs.py`,
`tests/fixtures/benchmark-workloads.json`, and `docs/BENCHMARKING.md`.
Dependency: N1 schema frozen; can run alongside N2.

Keep the existing default CLI/report behavior compatible. Add a versioned
workload file with short chat, longer document input, multi-turn history and
structured-output requests. Retain warmup, repeated and unrelated groups and
all failures. Capture truncation/finish reasons and reasoning time separately
from first answer content. Store workload hashes and explicit session identity.

Compare matched runs: medians, ranges and sample counts for client latency,
server-reported prefill/decode and success rate. Preserve metric definitions;
do not manufacture decode rates from characters or wall time. Refuse or label
mismatches in model, template, configuration, workload, hardware or build.
Prompt sizes are bytes unless a trusted tokenizer/server supplies token counts.
Join diagnostic records using run identity and timestamps; detect missing or
truncated traces. An app restart is not proof of a cold filesystem cache.

Acceptance: mock streams cover old/new schemas, missing counters, partial runs,
early stops, incompatible comparisons and clock/session mismatches. Existing
deadline and redaction behavior remains intact. Codex checks arithmetic against
hand-calculated fixtures and runs the full benchmark regression suite.

### N4. Add a compact answer-quality regression suite

Scope: new `scripts/evaluate-ps5lm.py`, `tests/test_evaluate_ps5lm.py`,
`tests/fixtures/quality/`, and `docs/EVALUATION.md`. Dependency: N3 request/report
interface accepted. Separate assignment from the benchmark implementation.

Create 20-30 small, original cases covering factual extraction from supplied
text, arithmetic, schema-constrained JSON, instruction adherence, multi-turn
reference resolution and code understanding. Start with text-only inference.
Use deterministic validators where possible; retain raw output and per-case
failure reasons. Do not execute model-generated code. For free-form answers,
use a written manual rubric and blinded A/B labels; no model-as-judge dependency.

Include designated critical cases and context-stress variants. Fix and hash the
cases and rubrics before comparing configurations. Make the harness usable
against the same already-running endpoint and retain timeout/transport failures
separately from incorrect answers. This is a project regression suite, not proof
of general model intelligence; do not reduce all dimensions to one opaque score.

Acceptance: mock answers exercise passing, incorrect, malformed, truncated and
timed-out cases; graders cannot pass on superficial substrings. Codex independently
checks every fixture/rubric and manually reviews selected real outputs.

## Milestone B: select and implement one measured improvement

### N5. Capture a controlled console baseline

Scope: new `scripts/collect-ps5lm-run.py`, its mocked tests and
`docs/validation/console-protocol.md`; raw output under ignored `build/`.
Dependencies: N0, N2-N4. Worker builds collection/orchestration only; Codex runs
and reviews the actual console trials when connection details are available.

Require an explicit console target and run manifest. Collect existing logs and
new diagnostics, wait for readiness with a deadline, run speed/quality suites,
and preserve failures. Default to collection against the running model. Any
automated load/restart schedule must be explicitly selected; no deployment or
model download is part of this tool. Validate that the loaded configuration
matches the manifest before each trial.

Keep the first matrix small:

1. Current known-good model/configuration, collection off/on.
2. Same weights under `balanced` and `kv_precision` when they actually select
   different settings; record selected context/KV, not just profile names.
3. At most two already-available alternatives: a smaller model at higher weight
   precision and a larger model at lower precision, if both fit conservatively.
4. Matched context/KV comparisons to isolate weight/model changes, followed by
   practical comparisons at each model's useful configuration.

Use at least five measured repetitions per speed condition, alternate baseline
and candidate order, and record warmup separately. Run three load cycles for
the finalist and a 30-minute mixed-request stability session. Keep thermal
conditions, storage, template/thinking settings and output limits in the record.
Separate fresh process loads, warm loads and unknown filesystem cache state.
Repeat on PS5 Pro only if that hardware is available; otherwise leave Pro open.

Acceptance: raw records reproduce the summary and expose errors, actual tokens,
missing counters, memory trends and quality results. No console crash, device
loss or runaway restart cycle is acceptable. A short successful chat is not
substituted for the stability session.

### N6. Explain the memory plan without changing selection policy

Scope: `ps5/app/model_plan.hpp`, `model_plan.cpp`, `model_plan_test.cpp`, and
`docs/PLANNING.md`. Dependency: N2 measurement semantics accepted; may precede
the full N5 experiment. No settings/dashboard edits in this package.

Separate weights/projector, KV, recurrent state, checkpoint, compute estimate,
vision workspace and safety allowance in the plan. Keep an explicit measured,
derived or estimated provenance for each term. Preserve existing totals and
candidate choices initially, including checkpoint accounting; explain rather
than accidentally remove what looks like duplicate recurrent state. Use bytes
internally where practical and test overflow/boundary behavior.

Acceptance: existing selection fixtures unchanged; itemized terms reproduce
the old totals within a declared rounding tolerance; missing/uncertain metadata
does not become a confident fit. Codex checks the equations against the loader
and actual observations. This package alone does not claim better fit accuracy.

### N7. Calibrate one reserve or execution setting

Scope: planner and its tests, a small source-controlled measurement fixture and
`docs/PLANNING.md`. Dependencies: N5 results and N6. Codex selects ONE hypothesis
before assigning the worker; it is not an open-ended optimization task.

Candidate hypotheses: a better-supported reserve for one architecture, a lower
context/KV tradeoff, or a conservative text microbatch adjustment. Use observations
from calibration runs and validate on held-out runs/configurations. Do not lower
reserves solely to make the fit badge green. Unknown architectures keep a
conservative fallback. Keep wide vision/decision batches out of a text-only
tuning change. Do not change driver patches or dependency revisions here.

Acceptance: correct totals, valid launch arguments and all regression checks;
held-out console load/prefill/generation fits with documented headroom. Ship a
new default only after the decision gate below. If no hypothesis wins, deliver
the measurements and retain the working defaults.

### N8. Show useful recommendations and measured limitations

Scope: `ps5/app/dashboard.cpp`, `stats.hpp`/producer fields if needed,
`preview_main.cpp`, and relevant planning/model docs. Dependencies: accepted
N6-N7 interfaces. Only one writer to shared UI/stats files.

Show an understandable memory breakdown, effective context/KV, and whether a
configuration is estimated or console-tested. Keep runtime data separate from
catalog claims. Explain when a setting applies only on next load. Recommendations
must name the tested hardware/build/model quant and workload; do not relabel
`kv_precision` as better model quality. Keep unknown/unsupported states actionable.

Acceptance: preview fixtures for comfortable fit, near-limit fit, unsupported
bundle and unavailable telemetry; Codex visually reviews renders, then checks
controller navigation and actual selected launch arguments on console.

### N9. Package the validated result reproducibly

Scope: `scripts/release-app.sh`, one packaging test file, `docs/BUILDING.md`,
`docs/CI.md` and isolated CI changes if needed. Dependency: N8 accepted.

Promote the useful task-local packaging checks into maintained tooling: required
title assets, dependency/source identity, final binary hashes, UI asset provenance,
archive CRC/content checks and a checksum sidecar. Generate the validation record
after the final browser-enabled build so it describes the shipped executable.
Keep local development identifiers distinct from release versions. Record tool
versions and dependency pins; byte-identical rebuilds are a separate goal unless
actually demonstrated. Do not carry machine-specific chroot paths into tooling.

Acceptance: incomplete/mismatched package fixtures fail; a full native build and
final ELF/FSELF inspection pass; Codex independently verifies ZIP contents/hashes
and performs the console smoke check. Produce a local deployable artifact; no
publishing or uploading is implicit in this package.

## Decision gate for changing defaults

These are proposed project acceptance targets, not measured results:

- Quality: no new failure in designated critical cases. Prefer a configuration
  with fewer real task failures; use manual A/B evidence for ambiguous answers.
- Performance: for a speed-driven change, target at least 10% improvement in the
  relevant median latency or server rate, reproduced in two sessions. Report
  dispersion; do not promote a difference indistinguishable from run noise.
- Tradeoff: a quality-driven candidate may be slower. Report its latency/memory
  cost explicitly and retain the faster option rather than silently replacing it.
- Stability: pass the finalist load cycles and mixed-request session, with no
  OOM, device loss or progressive unexplained memory growth. Sampled memory
  readings alone do not certify the absence of transient allocation peaks.
- Scope: a result on Slim supports a Slim recommendation; it does not establish
  Pro behavior. No hardware-specific constants without hardware evidence.

## Subsequent work, kept outside this milestone

**Complete split-model downloads** is the next product feature if the winning
resident model requires shards. Break it into three sequential assignments:

1. A pure bundle-manifest validator (`model_bundle.*` and tests): immutable repo
   revision, full shard set, exact names/sizes/hashes, total storage budget,
   duplicates/missing parts/path validation. No network or queue changes.
2. Queue/persistence integration (`market.*` and mocked tests): staged bundle,
   per-file verified resume, cancellation and restart recovery. Preserve existing
   single-file queues. Do not expose a partially published model to the library;
   Codex chooses an atomic directory or explicit completion-marker protocol
   before implementation. Stripped expert bundles remain unsupported.
3. UI/library integration after the queue contract is accepted: aggregate progress,
   actionable errors and model visibility only after all shards pass validation.
   Fault-inject disconnect, hash mismatch, full disk and process interruption;
   then perform a complete console download/load test.

**SSD expert paging** remains a separate research milestone. First assign a
bounded native-title read probe (not a payload-only proxy): read-only user-selected
file, sequential and seeded random reads, fixed buffer/concurrency/byte/time caps,
cancellation, and raw latency/throughput distributions. Cached reads are labelled
as such; do not call them physical SSD throughput. Keep probe code isolated from
normal inference until reviewed.

Then compare measured I/O cost with the chosen model's routed bytes per token,
expected misses and shared-memory cache/staging budget. If the needed evidence is
missing, the decision stays open. Only a viable measured budget unlocks packages
7-9 of the [earlier plan](ps5lm-improvements.md): expert reader, bounded cache,
Vulkan transfer, routed execution and PLE integration. Each stays a separate
correctness-reviewed chunk. Do not begin a large-model port as part of N0-N9.

## Dispatch order

1. N0, then N1 to fix the diagnostic interface.
2. N2 and N3 in parallel with disjoint write scopes.
3. N4 after N3; N6 can run alongside N4 after N2 semantics are reviewed.
4. N5 tooling and Codex-run measurements; hold N7 until evidence exists.
5. N7, N8, then N9 sequentially, with Codex review at every handoff.

At most two implementation workers run together in this stage, leaving Codex
room for review. Add new checks to the host runner through a single integration
change. Run focused checks for each chunk and the complete host/native suite for
the integrated candidate. The immediate starting work is N0-N3; the first major
checkpoint is trustworthy comparative data, not an unmeasured model upgrade.
