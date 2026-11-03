// SPDX-License-Identifier: GPL-3.0-or-later
#include "probe.hpp"
#include "stats.hpp"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ps5lm;

static void test_pro_model_detection() {
    assert(is_pro_model("CFI-7015") == true);
    assert(is_pro_model("CFI-7019") == true);
    assert(is_pro_model("CFI-7100") == true);
    assert(is_pro_model("CFI-7102B") == true);

    assert(is_pro_model("CFI-2015") == false);
    assert(is_pro_model("CFI-1115A") == false);
    assert(is_pro_model("CFI-1000") == false);
    assert(is_pro_model("host") == false);
    assert(is_pro_model("") == false);
}

static void test_json_formatting_and_write() {
    ProbeReport rep;
    rep.console_model = "CFI-7015";
    rep.is_pro = true;
    rep.firmware_version = "0x08000001";
    rep.cpu_freq_mhz = 3500;
    rep.direct_memory_size_bytes = 14763950080ull;
    rep.direct_memory_avail_bytes = 13540000000ull;
    rep.max_contiguous_direct_alloc_bytes = 13421772800ull;

    rep.vulkan.device_name = "AMD Radeon RX 7000 Series (RADV)";
    rep.vulkan.subgroup_size = 64;
    rep.vulkan.int_dot_product_4x8_signed = true;
    rep.vulkan.int_dot_product_4x8_unsigned = true;
    rep.vulkan.int_dot_product_mixed = true;
    rep.vulkan.cooperative_matrix_supported = true;
    rep.vulkan.extensions = { "VK_KHR_shader_integer_dot_product", "VK_EXT_memory_budget" };
    rep.vulkan.gpu_copy_bandwidth_gbps = 425.8;

    VulkanHeapReport heap;
    heap.size_bytes = 14763950080ull;
    heap.budget_bytes = 13540000000ull;
    heap.usage_bytes = 1223950080ull;
    heap.flags = 1;
    rep.vulkan.heaps.push_back(heap);

    rep.cpu_bandwidth.buffer_mib = 256;
    rep.cpu_bandwidth.read_1thread_gbps = 33.4;
    rep.cpu_bandwidth.read_3threads_gbps = 79.1;

    const std::string json_str = format_probe_json(rep);
    assert(!json_str.empty());

    // Validate that it parses cleanly as JSON
    const nlohmann::json j = nlohmann::json::parse(json_str);
    assert(j.is_object());

    // Check top-level keys
    assert(j.contains("console_model") && j["console_model"] == "CFI-7015");
    assert(j.contains("is_pro") && j["is_pro"] == true);
    assert(j.contains("firmware_version") && j["firmware_version"] == "0x08000001");
    assert(j.contains("cpu_freq_mhz") && j["cpu_freq_mhz"] == 3500);
    assert(j.contains("direct_memory_size_bytes") && j["direct_memory_size_bytes"] == 14763950080ull);
    assert(j.contains("direct_memory_avail_bytes") && j["direct_memory_avail_bytes"] == 13540000000ull);
    assert(j.contains("max_contiguous_direct_alloc_bytes") && j["max_contiguous_direct_alloc_bytes"] == 13421772800ull);

    // Check vulkan object and keys
    assert(j.contains("vulkan") && j["vulkan"].is_object());
    const auto & vj = j["vulkan"];
    assert(vj.contains("device_name") && vj["device_name"] == "AMD Radeon RX 7000 Series (RADV)");
    assert(vj.contains("subgroup_size") && vj["subgroup_size"] == 64);
    assert(vj.contains("int_dot_product_4x8_signed") && vj["int_dot_product_4x8_signed"] == true);
    assert(vj.contains("int_dot_product_4x8_unsigned") && vj["int_dot_product_4x8_unsigned"] == true);
    assert(vj.contains("int_dot_product_mixed") && vj["int_dot_product_mixed"] == true);
    assert(vj.contains("cooperative_matrix_supported") && vj["cooperative_matrix_supported"] == true);
    assert(vj.contains("extensions") && vj["extensions"].is_array() && vj["extensions"].size() == 2);
    assert(vj.contains("gpu_copy_bandwidth_gbps") && vj["gpu_copy_bandwidth_gbps"].get<double>() > 400.0);
    assert(vj.contains("heaps") && vj["heaps"].is_array() && vj["heaps"].size() == 1);
    assert(vj["heaps"][0]["size_bytes"] == 14763950080ull);
    assert(vj["heaps"][0]["budget_bytes"] == 13540000000ull);
    assert(vj["heaps"][0]["usage_bytes"] == 1223950080ull);
    assert(vj["heaps"][0]["flags"] == 1);

    // Check cpu_bandwidth object and keys
    assert(j.contains("cpu_bandwidth") && j["cpu_bandwidth"].is_object());
    const auto & cj = j["cpu_bandwidth"];
    assert(cj.contains("buffer_mib") && cj["buffer_mib"] == 256);
    assert(cj.contains("read_1thread_gbps") && cj["read_1thread_gbps"].get<double>() > 30.0);
    assert(cj.contains("read_3threads_gbps") && cj["read_3threads_gbps"].get<double>() > 70.0);

    // Test write_probe_report
    const std::string test_path = "build/host-tests/test_probe.json";
    std::string err;
    assert(write_probe_report(test_path, rep, &err) && err.empty());

    // Verify written file exists and contains the exact content
    std::ifstream in(test_path);
    assert(in.is_open());
    const std::string read_back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    assert(read_back == json_str);
    in.close();

    std::filesystem::remove(test_path);
}

static void test_cpu_bandwidth_measurement() {
    // 4 MiB test buffer on host
    constexpr size_t kTestBytes = 4ULL * 1024 * 1024;
    CpuBandwidthReport rep = measure_cpu_bandwidth(kTestBytes);
    assert(rep.buffer_mib == 4);
    assert(rep.read_1thread_gbps > 0.0);
    assert(rep.read_3threads_gbps > 0.0);
}

static void test_stats_collector() {
    StatsCollector stats;
    Live model_part;
    model_part.model_label = "test_model";
    model_part.spec_mode = "mtp";
    stats.set_model(model_part);
    Live snap = stats.snapshot();
    assert(snap.model_label == "test_model");
    assert(snap.spec_mode == "mtp");
    assert(snap.draft_tokens == 0);
    assert(snap.draft_accepted == 0);
    assert(snap.draft_acceptance_rate == 0.0f);
}

int main() {
    test_pro_model_detection();
    test_json_formatting_and_write();
    test_cpu_bandwidth_measurement();
    test_stats_collector();
    std::printf("probe_test: ok (pro model detection, json serialization, atomic write, cpu bandwidth benchmark, stats collector)\n");
    return 0;
}
