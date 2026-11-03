#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

host_build="$repo_root/build/host-tests"
mkdir -p "$host_build" "$repo_root/build/pycache"
export PYTHONPYCACHEPREFIX="$repo_root/build/pycache"

echo "== model planner (C++17) =="
c++ -std=c++17 -O1 \
    ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp \
    -o "$host_build/model_plan_test"
"$host_build/model_plan_test"

echo "== model market integration (C++17, mocked HTTP) =="
gcc -std=c99 -O1 \
    -Ithird_party/llama.cpp/vendor/hash \
    -c third_party/llama.cpp/vendor/hash/sha256/sha256.c \
    -o "$host_build/market-test-sha256.o"
c++ -std=c++17 -O1 -ffunction-sections -fdata-sections \
    -DPS5LM_MARKET_TEST_QUEUE \
    -Ithird_party/llama.cpp/vendor \
    -Ithird_party/llama.cpp/vendor/hash \
    -Ips5/app \
    ps5/app/market_plan_test.cpp ps5/app/market.cpp ps5/app/model_plan.cpp \
    "$host_build/market-test-sha256.o" \
    -Wl,--gc-sections -pthread \
    -o "$host_build/market_plan_test"
"$host_build/market_plan_test"

echo "== settings persistence (C++20) =="
c++ -std=c++20 -Ithird_party/llama.cpp/vendor \
    ps5/app/settings_test.cpp ps5/app/settings.cpp \
    -o "$host_build/settings_test"
"$host_build/settings_test"

echo "== console probe (C++20) =="
c++ -std=c++20 -O1 -pthread \
    -Ithird_party/llama.cpp/vendor \
    -Ips5/app \
    ps5/app/probe_test.cpp ps5/app/probe.cpp ps5/app/stats.cpp \
    -o "$host_build/probe_test"
"$host_build/probe_test"

echo "== dashboard host syntax (C++20) =="
c++ -std=c++20 -fsyntax-only \
    -Ithird_party/ps5-homebrew-ui/src \
    -Ithird_party/ps5-homebrew-ui/host \
    -Ithird_party/ps5-homebrew-ui/third_party \
    -Ips5/app ps5/app/dashboard.cpp

echo "== bounded expert source (C++17) =="
c++ -std=c++17 -O1 -Wall -Wextra -Werror -DPS5LM_MOE_EXPERT_SOURCE_TESTING \
    ps5/moe/expert_source_test.cpp ps5/moe/expert_source.cpp \
    -o "$host_build/expert_source_test"
"$host_build/expert_source_test"

echo "== transactional expert cache (C++17) =="
c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic \
    ps5/moe/slot_cache.cpp ps5/moe/slot_cache_test.cpp \
    -o "$host_build/slot_cache_test"
"$host_build/slot_cache_test"

echo "== per-layer expert paging runtime (C++17) =="
c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic \
    ps5/moe/expert_source.cpp ps5/moe/slot_cache.cpp ps5/moe/paging_runtime.cpp ps5/moe/paging_runtime_test.cpp \
    -o "$host_build/paging_runtime_test"
"$host_build/paging_runtime_test"

echo "== bounded storage probe (C/C++) =="
make -C probes/storageprobe MODE=host host-test

echo "== Python build queue regressions =="
python3 -m unittest discover -s tests -p 'test_build_app.py' -v

echo "== Python streaming benchmark regressions =="
python3 -m unittest discover -s tests -p 'test_benchmark_ps5lm.py' -v

echo "== GGUF header audit =="
python3 -m unittest discover -s tests -p 'test_inspect_gguf_header.py' -v

echo "Host regression tests passed. Build output: build/host-tests"
