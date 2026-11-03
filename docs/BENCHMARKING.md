# HTTP streaming benchmark

scripts/benchmark-ps5lm.py measures client-visible request timings against an
already-running OpenAI-compatible streaming chat endpoint. It does not start a
server, contact a console by itself, load a model, or download model files.

Use the full chat completions URL and save the raw JSON report under ignored
build/ output:

~~~powershell
python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --model default --warmup --repeats 5 --max-tokens 256 --timeout 60 --output build/ps5lm-benchmark.json
~~~

--url is required. --model defaults to default; pass the label accepted by the
server if it requires a particular value. If the server requires bearer
authentication, explicitly pass --api-key-env NAME to read a bearer token
from that environment variable. Authentication is off by default, so an
OPENAI_API_KEY already present in the shell is not sent to the endpoint. The
key itself is never written to the report. URL query values are redacted in
the saved endpoint identity.

## Trial schedule and settings

The client runs one optional short warmup prompt first, then a fixed technical
explanation prompt --repeats times, followed by three fixed, unrelated
explanation prompts. The measured prompts ask for about 100 to 120 words to
exercise more than first-token latency. Prompt IDs and text are included in
every trial. Warmup is labeled warmup and stays visible in the report and
summary; it is not mixed into the repeated or unrelated groups. No prompts are
randomized. A model may stop early regardless of the requested length; use
server-reported predicted_n, when available, to see the actual generated token
count.

Each request uses streaming, temperature: 0, top_p: 1, seed: 1, zero presence
and frequency penalties, and max_tokens: 256 by default. Override the output
limit with --max-tokens. The client requests server usage with
stream_options.include_usage: true and sends llama.cpp's cache_prompt: false
extension to avoid prompt-cache reuse between requests. These values and the
client version, Python version, endpoint, requested model, timeout, run
timestamp, response model IDs and available server headers are recorded in the
report. The response model is read from the stream when provided; the client
does not require a separate model-list or health endpoint.

The default timeout is 60 seconds per trial. A socket deadline bounds request
headers and streaming through the [DONE] event, including trickling streams,
rather than resetting for every socket read. Use the console's IP address:
hostname resolution uses the operating system resolver and cannot be
interrupted by the socket watchdog. If a trial fails, the client retains its
partial measurements and error in the JSON, continues the schedule, and exits
with status 1 after writing the report. An output-file error exits with status
2.

## Reading the measurements

Each trial reports:

- first_content_ms: elapsed client time from trial start until the first
  non-empty assistant delta.content. Role-only and reasoning-only deltas do
  not count. first_role_ms and first_reasoning_ms are reported separately
  when those deltas are present. Since the clock starts before connecting,
  this includes connection setup and response headers.
- wall_time_ms: elapsed client time from trial start through receipt of
  [DONE], including connection setup and response handling.
- server_reported.usage and server_reported.timings: the server's fields
  copied from stream events, or null when absent. The Server-Timing header
  is also retained verbatim when present.
- model_identity, response ID, finish reason, output text, HTTP status, and
  errors when available.

The client does not estimate tokens from words or characters, derive a
tokens-per-second figure, or claim that a server timing field is a client
measurement. Do not treat client first-content or wall time as decode speed.
Missing usage stays null.

The report cannot establish model load time, physical reads or bytes, memory
use, peak allocations, CPU/GPU load, temperature, firmware, console identity,
or storage identity. Collect those separately with hardware instrumentation
and record the PS5 model, firmware, server build/commit, model file and
quantization, and storage device alongside the JSON. Compare runs only when
those conditions and the request settings are controlled.

The mock-server suite exercises fragmented SSE, [DONE], SSE and HTTP errors,
missing usage, stream truncation, idle and overall timeouts, warmup separation,
and preservation of failed trials:

~~~powershell
python -B tests/test_benchmark_ps5lm.py
~~~

## Speculative decoding benchmarks (needs console)

To measure speculative decoding on the console, run comparisons with the draft
parameters varied across code and prose prompt sets. Server-reported draft
tokens and accepted tokens are recorded in each trial's `server_reported`:

1. **Qwen 3.5 9B Baseline vs Companion Draft Model (Qwen 3.5 0.8B):**
   - **Baseline (Speculative Off):** Set `Speculative decoding` to `Off` in Settings.
     Run benchmark on both code and prose:
     ```powershell
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set code --warmup --repeats 5 --output build/bench-qwen9b-base-code.json
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set prose --warmup --repeats 5 --output build/bench-qwen9b-base-prose.json
     ```
   - **With Companion Draft (`draft_max` 4, 8, 16):** Set `Speculative decoding` to `Draft model`, pick Qwen 3.5 0.8B, and test draft_max values in `/data/PS5LM/settings.json` or CLI.
     ```powershell
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set code --warmup --repeats 5 --output build/bench-qwen9b-draft8-code.json
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set prose --warmup --repeats 5 --output build/bench-qwen9b-draft8-prose.json
     ```

2. **Qwen 3.8 27B Baseline vs Multi-Token Prediction (MTP):**
   - **Baseline (Speculative Off):**
     ```powershell
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set code --warmup --repeats 5 --output build/bench-qwen27b-base-code.json
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set prose --warmup --repeats 5 --output build/bench-qwen27b-base-prose.json
     ```
   - **With MTP (`--spec-type draft-mtp`, `draft_max` 4, 8, 16):** Set `Speculative decoding` to `Auto (MTP)` in Settings.
     ```powershell
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set code --warmup --repeats 5 --output build/bench-qwen27b-mtp8-code.json
     python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --prompt-set prose --warmup --repeats 5 --output build/bench-qwen27b-mtp8-prose.json
     ```

Record the following table across runs:
| Model & Mode | Prompt Set | draft_max | Generation tok/s | Acceptance Rate (%) | Peak Free Pool (GiB) |
|---|---|---|---|---|---|
| Qwen 3.5 9B (Base) | Code | N/A | | N/A | |
| Qwen 3.5 9B (Base) | Prose | N/A | | N/A | |
| Qwen 3.5 9B + 0.8B | Code | 4 / 8 / 16 | | | |
| Qwen 3.5 9B + 0.8B | Prose | 4 / 8 / 16 | | | |
| Qwen 3.8 27B (Base) | Code | N/A | | N/A | |
| Qwen 3.8 27B (Base) | Prose | N/A | | N/A | |
| Qwen 3.8 27B + MTP | Code | 4 / 8 / 16 | | | |
| Qwen 3.8 27B + MTP | Prose | 4 / 8 / 16 | | | |

## Integer dot product benchmarks (needs console)

To test the `RADV_PS5_INT_DOT` driver patch (`patches/mesa/0010-radv-ps5-report-integer-dot-product.patch`) on the console:

### 1. Configuration options in `/data/PS5LM/env.txt`

The refined driver patch supports three modes for `RADV_PS5_INT_DOT`:
- **`RADV_PS5_INT_DOT=auto` (Recommended for Pro)**: Automatically enables integer dot product on PS5 Pro (`CFI-70xx` / Viola), where RDNA 3 hardware contains working `v_dot4_*` instructions. On Base/Slim consoles (`Oberon` GC 10.1.3), it safely remains disabled.
- **`RADV_PS5_INT_DOT=1` (Force on)**: Forces integer dot product acceleration on any console. On Base/Slim consoles, RADV logs a warning to stderr:
  ```text
  radv/ps5: warning: RADV_PS5_INT_DOT=1 forced on non-Pro console (GC 10.1.3 omits v_dot4_* in hardware)
  ```
  Note: ggml-vulkan compute shaders (`MESA_SHADER_COMPUTE`) are isolated from NGG workgroup repack (which broke geometry shaders on GC 10.1.3 when dot products were reported), but testing on GC 10.1.3 is experimental.
- **`RADV_PS5_INT_DOT=0` (or unset)**: Disabled on all models (default behavior).
- **`RADV_PS5_PRO=1` (Override)**: Simulates PS5 Pro GPU topology (60 active CUs, 2.35 GHz boost, 2.25 GHz memory / 576 GB/s) for testing on host or console.

### 2. Verify device detection and initialization

Add the desired switch to `/data/PS5LM/env.txt` and launch the app.
Check `/data/PS5LM/llama.log` for driver and ggml-vulkan initialization:

**On PS5 Pro (`RADV_PS5_INT_DOT=auto` or `1`):**
```text
radv/ps5: integer dot product enabled (RADV_PS5_INT_DOT=auto)
ggml_vulkan: 0 = PlayStation 5 Pro GPU (radv) | ... | int dot: 1 | ...
```

**On Base / Slim (`RADV_PS5_INT_DOT=1` forced):**
```text
radv/ps5: warning: RADV_PS5_INT_DOT=1 forced on non-Pro console (GC 10.1.3 omits v_dot4_* in hardware)
radv/ps5: integer dot product enabled (RADV_PS5_INT_DOT=1)
ggml_vulkan: 0 = PlayStation 5 GPU (radv) | ... | int dot: 1 | ...
```

### 3. Correctness verification

Before benchmarking throughput, verify numerical stability and logit fidelity:
- Compare greedy sampling output (temperature 0, fixed seed) on a 512-token prompt with `RADV_PS5_INT_DOT=0` vs `RADV_PS5_INT_DOT=1` (or `auto` on Pro):
  ```powershell
  python scripts/benchmark-ps5lm.py --url http://PS5_IP:8081/v1/chat/completions --max-tokens 512 --repeats 1 --output build/intdot-verify-on.json
  ```
- Compare greedy output text and inspect `/v1/introspect` top-k logit rankings; tokens should match or differ only within expected floating-point accumulation variance.

### 4. Performance comparison (Prompt vs Decode tok/s)

Measure prompt processing throughput (2,000-token prompt) and generation tok/s across models with the switch disabled vs enabled:
- **Qwen 3.5 9B Q4_K_M** (flag off vs on)
- **Qwen 3.5 9B Q4_0** (flag off vs on: Q4_0 heavily exercises `mul_mat_vec_q4_0_q8_0` integer dot kernels)
- **Qwen 3.8 27B** (flag off vs on)

Record the results across runs:
| Console Model | Model & Quant | RADV_PS5_INT_DOT | Prompt tok/s (2k prompt) | Generation tok/s | Status / Anomaly |
|---|---|---|---|---|---|
| PS5 Slim (Oberon) | Qwen 3.5 9B Q4_K_M | 0 (default) | *(needs console)* | *(needs console)* | Normal baseline |
| PS5 Slim (Oberon) | Qwen 3.5 9B Q4_K_M | 1 (forced) | *(needs console)* | *(needs console)* | |
| PS5 Slim (Oberon) | Qwen 3.5 9B Q4_0 | 0 (default) | *(needs console)* | *(needs console)* | Normal baseline |
| PS5 Slim (Oberon) | Qwen 3.5 9B Q4_0 | 1 (forced) | *(needs console)* | *(needs console)* | |
| PS5 Slim (Oberon) | Qwen 3.8 27B | 0 (default) | *(needs console)* | *(needs console)* | Normal baseline |
| PS5 Slim (Oberon) | Qwen 3.8 27B | 1 (forced) | *(needs console)* | *(needs console)* | |
| PS5 Pro (Viola) | Qwen 3.5 9B Q4_K_M | auto | *(needs PS5 Pro)* | *(needs PS5 Pro)* | |
| PS5 Pro (Viola) | Qwen 3.5 9B Q4_0 | auto | *(needs PS5 Pro)* | *(needs PS5 Pro)* | |
| PS5 Pro (Viola) | Qwen 3.8 27B | auto | *(needs PS5 Pro)* | *(needs PS5 Pro)* | |

### 5. Failure modes to watch

- `VK_ERROR_DEVICE_LOST` during shader compilation or execution.
- Console freeze or GPU hang (if encountered, power cycle the console and leave the flag off).
- Output text corruption or NaN/divergent logits.
If any failure mode occurs, keep `RADV_PS5_INT_DOT` off and capture `/data/PS5LM/llama.log`.


