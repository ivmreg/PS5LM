#!/usr/bin/env bash
# Build and run the patched llama.cpp MoE paging CPU tests in an isolated copy.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

die() {
    printf 'moe-engine: %s\n' "$*" >&2
    exit 2
}

for tool in awk bash cmake git mkdir mktemp rm sed tar uname; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done

cc="${CC:-cc}"
cxx="${CXX:-c++}"
command -v "$cc" >/dev/null 2>&1 || die "missing C compiler: $cc"
command -v "$cxx" >/dev/null 2>&1 || die "missing C++ compiler: $cxx"

case "$(uname -s)" in
    Linux*|Darwin*) ;;
    *) die 'requires a POSIX host (Linux, macOS, or Linux under WSL)' ;;
esac

llama_submodule="$repo_root/third_party/llama.cpp"
llama_revision="$(git -C "$repo_root" ls-tree HEAD -- third_party/llama.cpp | awk '$2 == "commit" { print $3 }')"
[[ -n "$llama_revision" ]] || die 'third_party/llama.cpp is not pinned as a git submodule at HEAD'
[[ -f "$llama_submodule/CMakeLists.txt" ]] || die 'initialize third_party/llama.cpp first (see docs/CI.md)'
git -C "$llama_submodule" cat-file -e "$llama_revision^{commit}" 2>/dev/null || \
    die "pinned llama.cpp revision $llama_revision is unavailable in the initialized submodule"

fixture="$repo_root/tests/fixtures/moe_paging/qwen3moe-tiny-f32.gguf"
[[ -s "$fixture" ]] || die "missing checked-in tiny GGUF fixture: $fixture"

patches=("$repo_root"/patches/[0-9][0-9][0-9][0-9]-*.patch)
[[ -e "${patches[0]}" ]] || die 'no numbered patches found under patches/'

scratch_parent="${PS5LM_MOE_ENGINE_TMPDIR:-$repo_root/build}"
mkdir -p "$scratch_parent"
scratch_dir="$(mktemp -d "$scratch_parent/ps5lm-moe-engine.XXXXXX")"
cleanup() {
    status=$?
    trap - EXIT
    if [[ $status -eq 0 && "${PS5LM_MOE_ENGINE_KEEP_BUILD:-0}" != 1 ]]; then
        rm -rf -- "$scratch_dir"
    else
        printf 'moe-engine: scratch source and build retained at %s\n' "$scratch_dir" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

source_dir="$scratch_dir/llama.cpp"
build_dir="$scratch_dir/build"
mkdir -p "$source_dir"

echo "== archive pinned llama.cpp $llama_revision into isolated source =="
git -C "$llama_submodule" archive --format=tar "$llama_revision" | tar -xf - -C "$source_dir"
git -C "$source_dir" init -q

for patch in "${patches[@]}"; do
    patch_lf="$scratch_dir/$(basename -- "$patch")"
    sed 's/\r$//' "$patch" > "$patch_lf"
    echo "== apply $(basename -- "$patch") =="
    git -C "$source_dir" apply --check "$patch_lf"
    git -C "$source_dir" apply "$patch_lf"
done

default_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)"
jobs="${BUILD_JOBS:-$default_jobs}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die 'BUILD_JOBS must be a positive integer'
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"

echo '== configure patched llama.cpp for host CPU tests =='
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_COMMIT="$llama_revision" \
    -DGGML_CPU=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_EXAMPLES=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_HIP=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_SYCL=OFF \
    -DGGML_RPC=OFF \
    -DPS5LM_MOE_SOURCE_DIR="$repo_root/ps5/moe"

targets=(
    test-backend-sched-copy-callback
    test-moe-paged-loader
    test-moe-paging-inference
    test-moe-cache-slot-map
)
echo "== build MoE engine tests (BUILD_JOBS=$jobs) =="
cmake --build "$build_dir" --target "${targets[@]}"

export PS5LM_FIXTURE_PATH="$fixture"
for target in "${targets[@]}"; do
    echo "== run $target =="
    "$build_dir/bin/$target"
done

echo 'MoE engine CPU tests passed.'
