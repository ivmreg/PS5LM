# Planning profiles

PS5LM chooses a supported context and KV cache type from the available memory
estimate, the user's context limit, any model preset limit, the model's trained
context, and an optional forced KV type. Profiles change only how those
runtime candidates are ordered or capped. They do not change GGUF weights or
weight quantization.

The planner considers contexts in this descending set: 131072, 65536, 32768,
16384, 8192, and 4096 tokens. A user limit, preset limit, model context limit,
or decision-model limit can remove candidates above the effective cap. The
planner uses the first candidate in a profile's ordering that fits its memory
estimate. If none fits, it reports the least-memory candidate available under
that profile and returns no launch arguments.

## Profiles

- **`long_context`** is the default, including for settings files without a
  profile and for unknown profile values. It scans contexts from longest to
  shortest, and at each context scans `f16`, `q8_0`, then `q4_0`. Automatic
  `f16` candidates are skipped above 8192 tokens, preserving the prior policy.
- **`balanced`** uses the same cache ordering and automatic `f16` cutoff as
  `long_context`, but caps the effective context target at 16384 tokens. This
  prefers a moderate context; it does not guarantee answer quality or fit every
  workload better.
- **`kv_precision`** scans cache types in this order: `f16`, `q8_0`, then
  `q4_0`; for each type it scans contexts from longest to shortest. Automatic
  `f16` is allowed above 8192 tokens, so this profile can choose a shorter
  context to retain a higher-precision KV cache.

In all profiles, a forced KV type filters candidates to that type. Explicitly
forcing `f16` may use it above 8192 tokens in any profile. User, preset, model,
and decision-model context caps still apply. KV precision describes only the
key/value cache representation. It does not claim model-weight precision or
guarantee better answers.

## When changes take effect

The selected profile is persisted as `planning_profile` in
`/data/PS5LM/settings.json`. Changing it updates the local model plans. The
next model load uses the new profile to choose launch arguments. An
already-running model keeps the context and cache configuration with which it
was loaded until it is loaded again.

## Host checks

From the repository root, build the planner and settings checks with:

```sh
mkdir -p build
c++ -std=c++17 -O1 ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp -o build/model_plan_test
build/model_plan_test
c++ -std=c++20 -Ithird_party/llama.cpp/vendor ps5/app/settings_test.cpp ps5/app/settings.cpp -o build/settings_test
build/settings_test
```
