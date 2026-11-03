// SPDX-License-Identifier: GPL-3.0-or-later
#include "probe.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __PROSPERO__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>

#include <ps5platform/heap.h>
#include <ps5platform/kernel.h>
#include <vulkan/vulkan_core.h>

#include "ggml-backend.h"

extern "C" {
__attribute__((weak)) int sceKernelGetHwModelName(char * name);
long   sceKernelGetCpuFrequency(void);
size_t sceKernelGetDirectMemorySize(void);
int    sceKernelAvailableDirectMemorySize(off_t search_start, off_t search_end, size_t alignment,
                                          int64_t * phys_out, size_t * size_out);
int    sceKernelAllocateDirectMemory(off_t search_start, off_t search_end, size_t len,
                                     size_t alignment, int memory_type, off_t * phys_out);
int    sceKernelMapDirectMemory(void ** addr, size_t len, int prot, int flags,
                                off_t direct_start, size_t alignment);
int    sceKernelReleaseDirectMemory(off_t start, size_t len);
int    sceKernelMunmap(void * addr, size_t len);

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char * name);
}

#ifndef VK_KHR_shader_integer_dot_product
#define VK_KHR_shader_integer_dot_product 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR ((VkStructureType) 1000280001)
typedef struct VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR {
    VkStructureType sType;
    void *          pNext;
    VkBool32        integerDotProduct8BitUnsignedAccelerated;
    VkBool32        integerDotProduct8BitSignedAccelerated;
    VkBool32        integerDotProduct8BitMixedSignednessAccelerated;
    VkBool32        integerDotProduct4x8BitPackedUnsignedAccelerated;
    VkBool32        integerDotProduct4x8BitPackedSignedAccelerated;
    VkBool32        integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    VkBool32        integerDotProduct16BitUnsignedAccelerated;
    VkBool32        integerDotProduct16BitSignedAccelerated;
    VkBool32        integerDotProduct16BitMixedSignednessAccelerated;
    VkBool32        integerDotProduct32BitUnsignedAccelerated;
    VkBool32        integerDotProduct32BitSignedAccelerated;
    VkBool32        integerDotProduct32BitMixedSignednessAccelerated;
    VkBool32        integerDotProduct64BitUnsignedAccelerated;
    VkBool32        integerDotProduct64BitSignedAccelerated;
    VkBool32        integerDotProduct64BitMixedSignednessAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating8BitSignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating16BitSignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating32BitSignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating64BitSignedAccelerated;
    VkBool32        integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;
} VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR;
#endif

#ifndef VK_EXT_memory_budget
#define VK_EXT_memory_budget 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT ((VkStructureType) 1000237000)
typedef struct VkPhysicalDeviceMemoryBudgetPropertiesEXT {
    VkStructureType sType;
    void *          pNext;
    VkDeviceSize    heapBudget[VK_MAX_MEMORY_HEAPS];
    VkDeviceSize    heapUsage[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryBudgetPropertiesEXT;
#endif

#endif

namespace ps5lm {

namespace {

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

struct BwJob {
    const unsigned char * base = nullptr;
    size_t                len  = 0;
    uint64_t              sink = 0;
};

void * bw_reader(void * arg) {
    BwJob * j = static_cast<BwJob *>(arg);
#if defined(__AVX2__)
    __m256i         acc = _mm256_setzero_si256();
    const __m256i * v   = reinterpret_cast<const __m256i *>(j->base);
    const size_t    n   = j->len / sizeof(__m256i);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < n; i += 4) {
            acc = _mm256_add_epi64(acc, _mm256_load_si256(v + i));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(v + i + 1));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(v + i + 2));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(v + i + 3));
        }
    }
    uint64_t l[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(l), acc);
    j->sink = l[0] + l[1] + l[2] + l[3];
#else
    uint64_t         acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    const uint64_t * v    = reinterpret_cast<const uint64_t *>(j->base);
    const size_t     n    = j->len / sizeof(uint64_t);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < n; i += 4) {
            acc0 += v[i];
            acc1 += v[i + 1];
            acc2 += v[i + 2];
            acc3 += v[i + 3];
        }
    }
    j->sink = acc0 + acc1 + acc2 + acc3;
#endif
    return nullptr;
}

double run_bandwidth(unsigned char * buf, size_t len, int threads) {
    if (!buf || len == 0 || threads <= 0) {
        return 0.0;
    }
    pthread_t    th[3];
    BwJob        jobs[3];
    const int    nthreads = std::min(threads, 3);
    const size_t slice    = (len / nthreads) & ~(size_t) 127;
    const double t0       = now_s();
    for (int i = 0; i < nthreads; ++i) {
        jobs[i] = { buf + i * slice, slice, 0 };
        pthread_create(&th[i], nullptr, bw_reader, &jobs[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        pthread_join(th[i], nullptr);
    }
    const double dt = now_s() - t0;
    if (dt <= 0.0) {
        return 0.0;
    }
    return (double) slice * nthreads * 2 / dt / 1e9;
}

#ifdef __PROSPERO__
void probe_vulkan(VulkanProbeReport & rep, const std::function<void(const char *)> & on_phase) {
    auto pfnCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!pfnCreateInstance) {
        return;
    }

    VkApplicationInfo app_info  = {};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "ps5lm-probe";
    app_info.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType                = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo     = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (pfnCreateInstance(&inst_info, nullptr, &instance) == VK_SUCCESS && instance) {
        auto pfnDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
        auto pfnEnumeratePhysicalDevices =
            reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
        auto pfnGetPhysicalDeviceProperties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
        auto pfnGetPhysicalDeviceProperties2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        auto pfnGetPhysicalDeviceMemoryProperties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties"));
        auto pfnGetPhysicalDeviceMemoryProperties2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2"));
        auto pfnEnumerateDeviceExtensionProperties =
            reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(vkGetInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));

        uint32_t pdev_count = 0;
        if (pfnEnumeratePhysicalDevices && pfnEnumeratePhysicalDevices(instance, &pdev_count, nullptr) == VK_SUCCESS &&
            pdev_count > 0) {
                std::vector<VkPhysicalDevice> pdevs(pdev_count);
                if (pfnEnumeratePhysicalDevices(instance, &pdev_count, pdevs.data()) == VK_SUCCESS) {
                    VkPhysicalDevice pdev = pdevs[0];

                    if (pfnGetPhysicalDeviceProperties) {
                        VkPhysicalDeviceProperties props = {};
                        pfnGetPhysicalDeviceProperties(pdev, &props);
                        rep.device_name = props.deviceName;
                    }

                    if (pfnEnumerateDeviceExtensionProperties) {
                        uint32_t ext_count = 0;
                        pfnEnumerateDeviceExtensionProperties(pdev, nullptr, &ext_count, nullptr);
                        if (ext_count > 0) {
                            std::vector<VkExtensionProperties> exts(ext_count);
                            pfnEnumerateDeviceExtensionProperties(pdev, nullptr, &ext_count, exts.data());
                            for (const auto & ext : exts) {
                                rep.extensions.push_back(ext.extensionName);
                                if (std::strcmp(ext.extensionName, "VK_KHR_cooperative_matrix") == 0 ||
                                    std::strcmp(ext.extensionName, "VK_NV_cooperative_matrix") == 0) {
                                    rep.cooperative_matrix_supported = true;
                                }
                            }
                        }
                    }

                    bool has_mem_budget = false;
                    for (const auto & e : rep.extensions) {
                        if (e == "VK_EXT_memory_budget") {
                            has_mem_budget = true;
                            break;
                        }
                    }

                    if (pfnGetPhysicalDeviceMemoryProperties) {
                        VkPhysicalDeviceMemoryProperties mem_props = {};
                        pfnGetPhysicalDeviceMemoryProperties(pdev, &mem_props);

                        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props = {};
                        budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

                        VkPhysicalDeviceMemoryProperties2 mem_props2 = {};
                        mem_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
                        mem_props2.pNext = &budget_props;

                        if (has_mem_budget && pfnGetPhysicalDeviceMemoryProperties2) {
                            pfnGetPhysicalDeviceMemoryProperties2(pdev, &mem_props2);
                        }

                        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
                            VulkanHeapReport heap;
                            heap.size_bytes = mem_props.memoryHeaps[i].size;
                            heap.flags      = mem_props.memoryHeaps[i].flags;
                            if (has_mem_budget && pfnGetPhysicalDeviceMemoryProperties2) {
                                heap.budget_bytes = budget_props.heapBudget[i];
                                heap.usage_bytes  = budget_props.heapUsage[i];
                            } else {
                                heap.budget_bytes = heap.size_bytes;
                                heap.usage_bytes  = 0;
                            }
                            rep.heaps.push_back(heap);
                        }
                    }

                    if (pfnGetPhysicalDeviceProperties2) {
                        VkPhysicalDeviceSubgroupProperties subgroup_props = {};
                        subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

                        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR dot_props = {};
                        dot_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
                        subgroup_props.pNext = &dot_props;

                        VkPhysicalDeviceProperties2 props2 = {};
                        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                        props2.pNext = &subgroup_props;

                        pfnGetPhysicalDeviceProperties2(pdev, &props2);

                        rep.subgroup_size = subgroup_props.subgroupSize;
                        rep.int_dot_product_4x8_signed = dot_props.integerDotProduct4x8BitPackedSignedAccelerated != 0;
                        rep.int_dot_product_4x8_unsigned =
                            dot_props.integerDotProduct4x8BitPackedUnsignedAccelerated != 0;
                        rep.int_dot_product_mixed =
                            (dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated != 0) ||
                            (dot_props.integerDotProduct8BitMixedSignednessAccelerated != 0);
                    }
                }
            }
            if (pfnDestroyInstance) {
                pfnDestroyInstance(instance, nullptr);
            }
        }
    }

    if (rep.device_name.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
            const auto         type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                rep.device_name = ggml_backend_dev_description(dev);
                if (rep.heaps.empty()) {
                    size_t free_b = 0, total_b = 0;
                    ggml_backend_dev_memory(dev, &free_b, &total_b);
                    VulkanHeapReport heap;
                    heap.size_bytes   = total_b;
                    heap.budget_bytes = free_b;
                    heap.usage_bytes  = total_b > free_b ? total_b - free_b : 0;
                    heap.flags        = 1;
                    rep.heaps.push_back(heap);
                }
                break;
            }
        }
    }

    if (on_phase) {
        on_phase("benchmarking GPU copy bandwidth");
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
        const auto         type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                constexpr size_t  kCopyBytes = 256ULL * 1024 * 1024;
                const size_t      nelems     = kCopyBytes / sizeof(float);
                ggml_init_params  p          = { 1024 * 1024, nullptr, true };
                ggml_context *    ctx        = ggml_init(p);
                if (ctx) {
                    ggml_tensor *         t_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelems);
                    ggml_tensor *         t_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelems);
                    ggml_backend_buffer_t b_src = ggml_backend_alloc_buffer(backend, ggml_nbytes(t_src));
                    ggml_backend_buffer_t b_dst = ggml_backend_alloc_buffer(backend, ggml_nbytes(t_dst));
                    if (b_src && b_dst) {
                        ggml_backend_tensor_alloc(b_src, t_src, nullptr);
                        ggml_backend_tensor_alloc(b_dst, t_dst, nullptr);
                        ggml_backend_buffer_clear(b_src, 1);
                        ggml_backend_buffer_clear(b_dst, 0);
                        ggml_backend_synchronize(backend);

                        // Warmup copy
                        ggml_backend_tensor_copy(t_src, t_dst);
                        ggml_backend_synchronize(backend);

                        constexpr int kPasses = 4;
                        const double  t0      = now_s();
                        for (int pass = 0; pass < kPasses; ++pass) {
                            ggml_backend_tensor_copy(t_src, t_dst);
                        }
                        ggml_backend_synchronize(backend);
                        const double dt = now_s() - t0;
                        if (dt > 0.0) {
                            // Effective memory bus bandwidth (read + write: 2 * bytes / dt)
                            rep.gpu_copy_bandwidth_gbps = (double) (kCopyBytes * kPasses * 2ULL) / dt / 1e9;
                        }
                        ggml_backend_buffer_free(b_src);
                        ggml_backend_buffer_free(b_dst);
                    } else {
                        if (b_src) {
                            ggml_backend_buffer_free(b_src);
                        }
                        if (b_dst) {
                            ggml_backend_buffer_free(b_dst);
                        }
                    }
                    ggml_free(ctx);
                }
                ggml_backend_free(backend);
            }
            break;
        }
    }
}
#endif

}  // namespace

bool is_pro_model(const std::string & model) {
    return model.rfind("CFI-7", 0) == 0;
}

CpuBandwidthReport measure_cpu_bandwidth(size_t buffer_bytes) {
    CpuBandwidthReport rep;
    rep.buffer_mib = buffer_bytes / (1024 * 1024);
    if (rep.buffer_mib == 0 && buffer_bytes > 0) {
        rep.buffer_mib = 1;
    }

    void * p           = nullptr;
    bool   from_direct = false;
#ifdef __PROSPERO__
    off_t  phys = 0;
    void * addr = (void *) 0x1000000000ULL;
    int    rc   = sceKernelAllocateDirectMemory(0, (off_t) sceKernelGetDirectMemorySize(), buffer_bytes, 2 * 1024 * 1024,
                                                12, &phys);
    if (rc == 0) {
        rc = sceKernelMapDirectMemory(&addr, buffer_bytes, 0x33, 0, phys, 2 * 1024 * 1024);
        if (rc == 0) {
            p           = addr;
            from_direct = true;
        } else {
            sceKernelReleaseDirectMemory(phys, buffer_bytes);
        }
    }
#endif
    if (!p) {
        if (posix_memalign(&p, 64, buffer_bytes) != 0) {
            p = nullptr;
        }
    }
    if (!p) {
        return rep;
    }

    std::memset(p, 1, buffer_bytes);
    rep.read_1thread_gbps  = run_bandwidth(reinterpret_cast<unsigned char *>(p), buffer_bytes, 1);
    rep.read_3threads_gbps = run_bandwidth(reinterpret_cast<unsigned char *>(p), buffer_bytes, 3);

#ifdef __PROSPERO__
    if (from_direct) {
        sceKernelMunmap(p, buffer_bytes);
        sceKernelReleaseDirectMemory(phys, buffer_bytes);
        p = nullptr;
    }
#endif
    if (p) {
        free(p);
    }
    return rep;
}

std::string format_probe_json(const ProbeReport & report) {
    nlohmann::json j;
    j["console_model"]                    = report.console_model;
    j["is_pro"]                           = report.is_pro;
    j["firmware_version"]                 = report.firmware_version;
    j["cpu_freq_mhz"]                     = report.cpu_freq_mhz;
    j["direct_memory_size_bytes"]         = report.direct_memory_size_bytes;
    j["direct_memory_avail_bytes"]        = report.direct_memory_avail_bytes;
    j["max_contiguous_direct_alloc_bytes"] = report.max_contiguous_direct_alloc_bytes;

    nlohmann::json vk;
    vk["device_name"] = report.vulkan.device_name;
    nlohmann::json heaps = nlohmann::json::array();
    for (const auto & h : report.vulkan.heaps) {
        nlohmann::json hj;
        hj["size_bytes"]   = h.size_bytes;
        hj["budget_bytes"] = h.budget_bytes;
        hj["usage_bytes"]  = h.usage_bytes;
        hj["flags"]        = h.flags;
        heaps.push_back(hj);
    }
    vk["heaps"]                        = heaps;
    vk["subgroup_size"]                = report.vulkan.subgroup_size;
    vk["int_dot_product_4x8_signed"]   = report.vulkan.int_dot_product_4x8_signed;
    vk["int_dot_product_4x8_unsigned"] = report.vulkan.int_dot_product_4x8_unsigned;
    vk["int_dot_product_mixed"]        = report.vulkan.int_dot_product_mixed;
    vk["cooperative_matrix_supported"] = report.vulkan.cooperative_matrix_supported;
    vk["extensions"]                   = report.vulkan.extensions;
    vk["gpu_copy_bandwidth_gbps"]      = report.vulkan.gpu_copy_bandwidth_gbps;
    j["vulkan"]                        = vk;

    nlohmann::json cpu;
    cpu["buffer_mib"]        = report.cpu_bandwidth.buffer_mib;
    cpu["read_1thread_gbps"]  = report.cpu_bandwidth.read_1thread_gbps;
    cpu["read_3threads_gbps"] = report.cpu_bandwidth.read_3threads_gbps;
    j["cpu_bandwidth"]        = cpu;

    return j.dump(2) + "\n";
}

bool write_probe_report(const std::string & path, const ProbeReport & report, std::string * error) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);

    const std::string tmp_path = path + ".tmp";
    FILE *            f        = std::fopen(tmp_path.c_str(), "wb");
    if (!f) {
        if (error) {
            *error = "cannot open " + tmp_path + ": " + std::strerror(errno);
        }
        return false;
    }
    const std::string json = format_probe_json(report);
    if (std::fwrite(json.data(), 1, json.size(), f) != json.size()) {
        if (error) {
            *error = "cannot write to " + tmp_path + ": " + std::strerror(errno);
        }
        std::fclose(f);
        unlink(tmp_path.c_str());
        return false;
    }
    std::fflush(f);
#if !defined(_WIN32)
    fsync(fileno(f));
#endif
    if (std::fclose(f) != 0) {
        if (error) {
            *error = "fclose failed on " + tmp_path + ": " + std::strerror(errno);
        }
        unlink(tmp_path.c_str());
        return false;
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        if (error) {
            *error = "rename " + tmp_path + " -> " + path + ": " + std::strerror(errno);
        }
        unlink(tmp_path.c_str());
        return false;
    }
    return true;
}

ProbeReport run_console_probe(std::function<void(const char *)> on_phase) {
    ProbeReport report;
#ifdef __PROSPERO__
    if (on_phase) {
        on_phase("reading hardware model and firmware");
    }
    char model_buf[256] = {};
    if (&sceKernelGetHwModelName != nullptr && sceKernelGetHwModelName(model_buf) == 0 && model_buf[0] != '\0') {
        report.console_model = model_buf;
    } else {
        report.console_model = "unknown";
    }
    report.is_pro = is_pro_model(report.console_model);

    uint32_t fw_val = 0;
    size_t   fw_len = sizeof(fw_val);
    if (sysctlbyname("kern.sdk_version", &fw_val, &fw_len, nullptr, 0) == 0 && fw_val != 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%08x", fw_val);
        report.firmware_version = buf;
    } else {
        report.firmware_version = "unknown";
    }

    const long hz = sceKernelGetCpuFrequency();
    if (hz > 0) {
        report.cpu_freq_mhz = (uint64_t) (hz / 1000000);
    }

    if (on_phase) {
        on_phase("querying direct memory");
    }
    const size_t pool = sceKernelGetDirectMemorySize();
    report.direct_memory_size_bytes = pool;

    int64_t at    = 0;
    size_t  avail = 0;
    if (sceKernelAvailableDirectMemorySize(0, pool, 2 * 1024 * 1024, &at, &avail) == 0) {
        report.direct_memory_avail_bytes = avail;
    }

    if (on_phase) {
        on_phase("measuring max contiguous direct memory allocation");
    }
    constexpr size_t kChunk256M = 256ULL * 1024 * 1024;
    constexpr size_t kAlign2M   = 2ULL * 1024 * 1024;
    size_t           start_len  = (report.direct_memory_avail_bytes / kChunk256M) * kChunk256M;
    if (start_len == 0 && report.direct_memory_avail_bytes >= kAlign2M) {
        start_len = (report.direct_memory_avail_bytes / kAlign2M) * kAlign2M;
    }

    for (size_t len = start_len; len >= kChunk256M; len -= kChunk256M) {
        off_t phys = 0;
        int   rc   = sceKernelAllocateDirectMemory(0, (off_t) pool, len, kAlign2M, 12, &phys);
        if (rc == 0) {
            void * addr = (void *) 0x1000000000ULL;
            rc          = sceKernelMapDirectMemory(&addr, len, 0x33, 0, phys, kAlign2M);
            if (rc == 0) {
                sceKernelMunmap(addr, len);
                sceKernelReleaseDirectMemory(phys, len);
                report.max_contiguous_direct_alloc_bytes = len;
                break;
            }
            sceKernelReleaseDirectMemory(phys, len);
        }
    }

    if (on_phase) {
        on_phase("probing Vulkan device");
    }
    probe_vulkan(report.vulkan, on_phase);
#else
    report.console_model                    = "host";
    report.is_pro                           = false;
    report.firmware_version                 = "unknown";
    report.cpu_freq_mhz                     = 0;
    report.direct_memory_size_bytes         = 0;
    report.direct_memory_avail_bytes        = 0;
    report.max_contiguous_direct_alloc_bytes = 0;
#endif

    if (on_phase) {
        on_phase("benchmarking CPU bandwidth (256 MiB)");
    }
    report.cpu_bandwidth = measure_cpu_bandwidth(256 * 1024 * 1024);

    if (on_phase) {
        on_phase("probe complete");
    }
    return report;
}

}  // namespace ps5lm
