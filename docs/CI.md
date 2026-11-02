# CI validation

[`native-build.yml`](../.github/workflows/native-build.yml) checks the existing app
on pull requests and pushes to `main` or `ci/**`. It uses ordinary Ubuntu hosted
runners, a read-only token, no secrets, and no console connection. Superseded runs
are cancelled. There are two independent jobs:

- **Host checks** (20-minute limit): shell/Python syntax, synthetic GGUF planner
  regressions, market split-model checks with mocked HTTP and FTP stubs,
  settings persistence, dashboard host syntax, and the Python build-queue and streaming-benchmark
  suites. It also builds and runs the pinned patch series' scheduler-copy,
  paged-loader, paging-inference and slot-map tests against the CPU backend. The workflow
  initializes the pinned llama.cpp and UI submodules first; the host test
  runners make no external network requests. Benchmark and market HTTP behavior
  uses local mocks only.
- **Full PS5 title cross-build** (45-minute limit, two compiler jobs): bootstrap
  the public SDK, reproduce clean-room `libc.prx`, build patched RADV and the
  existing llama-server/native dashboard, package the title folder, and inspect
  its ELF/FSELF structure and integrity. It does not execute the resulting title.

The second job downloads build tools, source dependencies and checksum-verified
public SDK archives. It does **not** download a trained model or proprietary
console firmware. Only the optional browser UI is omitted: its upstream fallback
fetches a floating `latest` release. The native console dashboard is built.
Normal `scripts/build-app.sh` builds keep the prebuilt browser UI enabled;
`PS5LM_PREBUILT_UI=OFF` explicitly opts out and clears legacy CMake aliases.

## Reproduce the checks

The existing host regression suites need Bash, Python 3, a C99 compiler, and a
C++17/C++20 compiler. The MoE engine CPU tests additionally need Git, tar, CMake
3.14 or newer, a C and C++17 compiler (GCC or Clang), and the selected CMake
generator's build tool (Make or Ninja). They require a POSIX host such as Linux,
macOS, or Linux under WSL (where packages can be installed via
`sudo apt update && sudo apt install -y cmake build-essential git`, or `uv tool install cmake`).
No Vulkan SDK or GPU is needed.

Initialize the pinned headers used by the market and settings tests, then run
the regression script from the repository root:

~~~sh
git submodule update --init --depth 1 -- third_party/llama.cpp third_party/ps5-homebrew-ui
for script in scripts/*.sh ps5/app/ps5cc ps5/app/ps5cxx; do bash -n "$script"; done
PYTHONPYCACHEPREFIX=build/pycache python3 -m py_compile scripts/*.py
bash scripts/test-host.sh
~~~

The MoE engine runner uses the pinned llama.cpp submodule and the checked-in
153 KiB synthetic Qwen3MoE fixture:

~~~sh
git submodule update --init --depth 1 -- third_party/llama.cpp
BUILD_JOBS=2 bash scripts/test-moe-engine.sh
~~~

It verifies the submodule gitlink revision, archives that commit into a unique
directory under `build/`, applies numbered patches 0001-0012 to that isolated
source copy, configures CPU-only CMake targets, and builds/runs
`test-backend-sched-copy-callback`, `test-moe-paged-loader`,
`test-moe-paging-inference`, and `test-moe-cache-slot-map`. The initialized submodule's working tree is not
patched or cleaned. The runner disables optional GPU backends, OpenMP, the
server/tools, and the prebuilt browser UI; it invokes the four binaries
directly, so the unrelated CTest entries that fetch models or vocabulary files
are not run. No model weights or network downloads are needed. Successful runs
remove their scratch directory; failures retain it for inspection. Set
`PS5LM_MOE_ENGINE_KEEP_BUILD=1` to keep a successful build, or
`PS5LM_MOE_ENGINE_TMPDIR` to place the scratch directory elsewhere.

The runner stores compiled checks in build/host-tests, Python bytecode in
build/pycache, and the mocked market queue in build/market-test-queue.json.
The market test's HTTP and FTP calls are replaced by stubs; it does not contact
Hugging Face or a console.

For the full build, use the digest-pinned Debian trixie container and exact apt
package list in the workflow, or an x86-64 Linux host with equivalent tools:

```sh
git submodule update --init --depth 1
export PATH=/usr/lib/llvm-19/bin:$PATH
LLVM_CONFIG=/usr/bin/llvm-config-19 BUILD_JOBS=2 scripts/test-native-build.sh
```

The script keeps dependencies in `.deps/native-ci-src` (override with
`PS5LM_NATIVE_SOURCES`) and validates existing dependency revisions before reuse.
It uses upstream's checked SDK setup and `make libc`, which rebuilds the runtime
twice and compares the recorded raw/signed hashes. That target also produces the
same host packaging tool required by the app, without building an unrelated demo
title. It applies this repository's three Mesa patches with fixed commit identity
and dates. The removed OpenGL release is fetched from the mirror already
documented in [BUILDING.md](BUILDING.md), checked against the original archive
hash, then checked by the pinned UI fetcher's file manifest.

| Input | Pin |
|---|---|
| Debian container | `trixie-20261020-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad` |
| PS5_Vulkan | `f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed` |
| PS5_PayloadSDK fork | `b83202be73e930050e00de0f1b4d9ec46c0391bf` |
| Mesa base | `1ba9874f478657ccf6da72ce55f0260164cc0bd0` |
| Mesa after this repository's patches | `9e0a7c63bc6d4b6bcb60831163bb3fe9f46e91ea` |
| llama.cpp / UI | The checkout's gitlink revisions (verified before building) |
| OpenGL SDK archive SHA-256 | `f93643c04c843d56143b00951df1f8042ea7706ae9f19e4e9abf158e1ead77c5` |

`build/native-ci/result.json` records the source pins and SHA-256 hashes of the
packaged files after successful inspections. The title is in
`build/app-ps5/title/PPSA99575`. A failed check exits nonzero; there is no automatic
deployment, release publication or binary execution. CI publishes logs, not a
release package.

## What a green result means

It proves the checked source can build/link/package with these public tools and
pass static ELF/FSELF checks, including the native UI and Vulkan driver build.
It does **not** prove the title boots on a PS5, the GPU executes inference, models
fit memory, or networking/performance works on hardware. A generic QEMU x86-64
guest cannot supply the PS5 loader, system services or AGC GPU interface; it would
not close that gap. Hardware smoke tests remain separate.

Source pins and archive checksums make the build inputs explicit, but apt package
updates and build paths/compiler versions can change title bytes. Only the
upstream clean-room runtime has a fixed byte-reproduction check; the title hashes
identify a particular build, not a promised reproducible release binary.
