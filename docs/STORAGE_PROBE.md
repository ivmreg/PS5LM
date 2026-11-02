# Bounded storage read probe

`probes/storageprobe` implements the P1 storage read probe for the beyond-RAM
model plan. It reads only a path the operator supplies, opens it read-only and
nonblocking, and refuses anything that is not a regular file before reading.
The nonblocking open keeps a FIFO from waiting for a writer. It runs sequential and seeded random reads at
4 KiB, 64 KiB, and 1 MiB. Random reads are aligned to their request size and
may repeat an offset; sequential reads are contiguous and start at a seeded
slot. The same seed and file layout produce the same offsets.

## Bounds and result meaning

The host CLI defaults to 16 samples per access-pattern/read-size group, a
64 MiB logical-byte limit, a 30 second elapsed-time limit, one synchronous read
at a time, one 1 MiB read buffer, and a fixed report structure (its exact byte
size is included in each JSON result). Hard limits are 32 samples per group,
80 MiB of logical bytes, and 60 seconds. The default schedule requests about 34.1 MiB
if the selected file is large enough. A smaller file lowers sequential sample
counts to avoid wrapping; groups whose read size exceeds the file are marked
`skipped_file_too_small`. The planned total is checked before the first read.

The JSON reports each `pread` offset, bytes returned, latency, and per-group
min/mean/p50/p95/max latency (nearest-rank percentiles). `logical_read_bytes`
counts bytes returned by `pread`. `physical_read_bytes` is always `null` and
`cache_state` is always `unknown`: this probe does not flush or inspect the OS,
console, or device cache, and it cannot claim physical SSD traffic. The
throughput fields are logical bytes per wall-clock second. A regular file can
also be sparse or changed by another process while the probe runs.

The elapsed-time cap is checked between system calls and during partial reads.
Each syscall requests at most 1 MiB. `SIGINT`, `SIGTERM`, or a title caller's
cancellation callback stops further reads; interrupted reads report the bytes
already returned. A blocked filesystem syscall cannot be forcibly stopped by
portable `pread` code, so the time cap is a work-loop bound rather than a hard
kernel-I/O deadline. Errors, short reads, cancellation, and time-limit exits
still produce a JSON report. The host CLI writes it to stdout; the title writes
it atomically to `/data/PS5LM/storage-probe.json`.

## Host fixture build and test

On Linux or macOS with a C compiler and `make`:

```sh
make -C probes/storageprobe MODE=host host-test
```

This builds the host CLI and test binary under ignored `build/storageprobe/`.
The deterministic test creates a small synthetic regular file, verifies
seeded offsets and read sizes, checks the read-only fixture remains unchanged,
exercises byte-budget refusal, cancellation, invalid file types and missing
files, and checks the JSON cache/accounting labels. It also rejects a FIFO
under a five second alarm and compiles a C++ caller against the C object to
verify the header's C linkage. It needs no PS5, model file, download, or
dependency change.

Example invocation against an explicitly selected existing file:

```sh
build/storageprobe/storageprobe-host --file /path/to/model.gguf --seed 1 --samples 16 --byte-limit-mib 64 --time-limit-sec 30
```

The report's `file_path` is included for attribution. Non-ASCII path bytes are
escaped byte-by-byte as JSON `\u00XX` values.

## Native-title request and result

`scripts/build-app.sh` compiles `storage_probe.c` into the PS5LM native title.
The app polls `/data/PS5LM/bench-read` with its existing request-file loop. Its
one-line content must be an absolute path to an existing regular file. The app
does not start a probe at launch. It starts one worker only for that explicit
request, uses the probe's bounded defaults, and rejects another request while
the worker is active. It writes the completed JSON to a temporary file, flushes
it, then renames it to `/data/PS5LM/storage-probe.json`. A new accepted run
clears the previous result so a fetch cannot mistake stale output for the
current run.

The worker has a 1 MiB stack, and the core uses one 1 MiB read buffer plus its
fixed report structure. Cancellation is checked between reads; a blocked
filesystem call can delay a user cancellation, but does not block rendering or
inference. Worker state and its cancellation context are process-owned. A model
restart or quit sets cancellation and calls `LoadExec` without joining the
worker, so a blocked read cannot hold the restart. If replacement interrupts
report writing, the temporary file is never renamed to the final result, and
the next app start removes that stale temporary file.

The controller script requires `PS5_HOST` as its target for every command. Run
the probe, fetch the result after completion, or cancel an active run explicitly:

```sh
PS5_HOST=192.168.1.50 scripts/ps5lm-app.sh bench-read '/data/PS5LM/models/model.gguf'
PS5_HOST=192.168.1.50 scripts/ps5lm-app.sh bench-read-result > build/storage-probe.json
PS5_HOST=192.168.1.50 scripts/ps5lm-app.sh bench-read-cancel
```

The host CLI reports `execution_context: "host"`; the integrated app writes
`execution_context: "native_title"`. Do not present host or payload readings
as native-title results. No deployment or console execution was performed for
this change.
