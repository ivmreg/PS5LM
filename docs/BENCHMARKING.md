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
