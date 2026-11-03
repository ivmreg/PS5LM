// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ps5lm {

struct VulkanHeapReport {
    uint64_t size_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t usage_bytes = 0;
    uint32_t flags = 0;
};

struct VulkanProbeReport {
    std::string device_name;
    std::vector<VulkanHeapReport> heaps;
    uint32_t subgroup_size = 0;
    bool int_dot_product_4x8_signed = false;
    bool int_dot_product_4x8_unsigned = false;
    bool int_dot_product_mixed = false;
    bool cooperative_matrix_supported = false;
    std::vector<std::string> extensions;
    // Effective bidirectional memory bus bandwidth for buffer-to-buffer copy (2 * size / dt, counting read + write).
    double gpu_copy_bandwidth_gbps = 0.0;
};

struct CpuBandwidthReport {
    size_t buffer_mib = 0;
    double read_1thread_gbps = 0.0;
    double read_3threads_gbps = 0.0;
};

struct ProbeReport {
    std::string console_model;
    bool is_pro = false;
    std::string firmware_version;
    uint64_t cpu_freq_mhz = 0;
    uint64_t direct_memory_size_bytes = 0;
    uint64_t direct_memory_avail_bytes = 0;
    uint64_t max_contiguous_direct_alloc_bytes = 0;
    VulkanProbeReport vulkan;
    CpuBandwidthReport cpu_bandwidth;
};

// Returns true if the hardware model string indicates a PS5 Pro (CFI-7xxx).
bool is_pro_model(const std::string & model);

// Measures CPU memory read bandwidth using 1 and 3 worker threads on a test buffer.
// Frees all allocated memory before returning.
CpuBandwidthReport measure_cpu_bandwidth(size_t buffer_bytes = 256 * 1024 * 1024);

// Produces deterministic, pretty-printed JSON from the probe report.
std::string format_probe_json(const ProbeReport & report);

// Atomically writes the probe report to disk via a .tmp file + fsync + rename.
bool write_probe_report(const std::string & path, const ProbeReport & report, std::string * error = nullptr);

// Runs the console probe sequence (memory, hardware, Vulkan, CPU/GPU bandwidth).
// Calls on_phase with progress descriptions if provided.
// Ensures all temporary allocations are freed before returning.
ProbeReport run_console_probe(std::function<void(const char *)> on_phase = {});

}  // namespace ps5lm
