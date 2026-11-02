// SPDX-License-Identifier: GPL-3.0-or-later
#include "model_plan.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

namespace ps5lm {

namespace {

constexpr double kGiB = 1073741824.0;
constexpr uint64_t kMaxHeaderBytes = 16ull << 20;
constexpr uint64_t kMaxMetadataItems = 1ull << 20;
constexpr uint64_t kMaxArrayItems = 1ull << 20;
constexpr uint32_t kMaxSplitCount = 4096;

constexpr uint64_t align_up(uint64_t offset, uint64_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

struct GgmlQuantTraits {
    uint32_t block_size;
    uint32_t type_size;
};

bool get_ggml_traits(uint32_t type, GgmlQuantTraits & traits) {
    switch (type) {
        case 0:  traits = { 1, 4 };   return true; // F32
        case 1:  traits = { 1, 2 };   return true; // F16
        case 2:  traits = { 32, 18 }; return true; // Q4_0
        case 3:  traits = { 32, 20 }; return true; // Q4_1
        case 6:  traits = { 32, 22 }; return true; // Q5_0
        case 7:  traits = { 32, 24 }; return true; // Q5_1
        case 8:  traits = { 32, 34 }; return true; // Q8_0
        case 9:  traits = { 32, 36 }; return true; // Q8_1
        case 10: traits = { 256, 84 }; return true; // Q2_K
        case 11: traits = { 256, 110 }; return true; // Q3_K
        case 12: traits = { 256, 144 }; return true; // Q4_K
        case 13: traits = { 256, 176 }; return true; // Q5_K
        case 14: traits = { 256, 210 }; return true; // Q6_K
        case 15: traits = { 256, 292 }; return true; // Q8_K
        case 16: traits = { 256, 66 }; return true; // IQ2_XXS
        case 17: traits = { 256, 74 }; return true; // IQ2_XS
        case 18: traits = { 256, 98 }; return true; // IQ3_XXS
        case 19: traits = { 256, 50 }; return true; // IQ1_S
        case 20: traits = { 32, 18 };  return true; // IQ4_NL
        case 21: traits = { 256, 110 }; return true; // IQ3_S
        case 22: traits = { 256, 82 }; return true; // IQ2_S
        case 23: traits = { 256, 136 }; return true; // IQ4_XS
        case 24: traits = { 1, 1 };   return true; // I8
        case 25: traits = { 1, 2 };   return true; // I16
        case 26: traits = { 1, 4 };   return true; // I32
        case 27: traits = { 1, 8 };   return true; // I64
        case 28: traits = { 1, 8 };   return true; // F64
        case 29: traits = { 256, 56 }; return true; // IQ1_M
        case 30: traits = { 1, 2 };   return true; // BF16
        case 34: traits = { 256, 54 }; return true; // TQ1_0
        case 35: traits = { 256, 66 }; return true; // TQ2_0
        case 39: traits = { 32, 17 };  return true; // MXFP4
        case 40: traits = { 64, 36 };  return true; // NVFP4
        case 41: traits = { 128, 18 }; return true; // Q1_0
        case 42: traits = { 64, 18 };  return true; // Q2_0
        default: return false;
    }
}

// GGUF metadata value types.
enum : uint32_t { U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 };

// A GGUF header from a file, or from memory (the start of a remote file);
// the PS5's libc has no fmemopen.
struct Reader {
    FILE *       f   = nullptr;
    const char * mem = nullptr;
    uint64_t     size = 0, pos = 0;
    bool         bad = false;

    bool can_read(uint64_t n) const {
        if (bad || pos > kMaxHeaderBytes || n > kMaxHeaderBytes - pos) {
            return false;
        }
        return f || (pos <= size && n <= size - pos);
    }

    size_t read(void * out, size_t n) {
        if (!can_read(n)) {
            bad = true;
            return 0;
        }
        if (n == 0) {
            return 0;
        }
        const size_t got = f ? std::fread(out, 1, n, f) : n;
        if (!f) {
            std::memcpy(out, mem + pos, n);
        }
        pos += got;
        if (got != n) {
            bad = true;
        }
        return got;
    }
    template <typename T> T get() {
        T v{};
        if (read(&v, sizeof(v)) != sizeof(v)) {
            bad = true;
        }
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (bad || n > kMaxHeaderBytes || n > (uint64_t) SIZE_MAX || !can_read(n)) {
            bad = true;
            return {};
        }
        std::string s((size_t) n, '\0');
        if (n && read(s.data(), n) != n) {
            bad = true;
        }
        return s;
    }
    void skip(uint64_t n) {
        if (!can_read(n)) {
            bad = true;
            return;
        }
        char discard[4096];
        while (n && !bad) {
            const size_t chunk = (size_t) std::min<uint64_t>(n, sizeof(discard));
            read(discard, chunk);
            n -= chunk;
        }
    }
};

uint64_t scalar_size(uint32_t t) {
    switch (t) {
        case U8: case I8: case BOOL: return 1;
        case U16: case I16: return 2;
        case U32: case I32: case F32: return 4;
        case U64: case I64: case F64: return 8;
        default: return 0;
    }
}

// A scalar as an integer; 0 for strings and arrays.
uint64_t read_int(Reader & r, uint32_t t) {
    switch (t) {
        case U8: return r.get<uint8_t>();
        case I8: return (uint64_t) r.get<int8_t>();
        case U16: return r.get<uint16_t>();
        case I16: return (uint64_t) r.get<int16_t>();
        case U32: return r.get<uint32_t>();
        case I32: return (uint64_t) r.get<int32_t>();
        case U64: return r.get<uint64_t>();
        case I64: return (uint64_t) r.get<int64_t>();
        case BOOL: {
            const uint8_t v = r.get<uint8_t>();
            if (v > 1) r.bad = true;
            return v;
        }
        case F32: {
            const float v = r.get<float>();
            if (!std::isfinite(v)) r.bad = true;
            return 0;
        }
        case F64: {
            const double v = r.get<double>();
            if (!std::isfinite(v)) r.bad = true;
            return 0;
        }
        default: r.bad = true; return 0;
    }
}

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

std::string basename(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string dirname(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0) return path.substr(0, 1);
    return path.substr(0, slash);
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir == ".") return dir.empty() ? name : dir + "/" + name;
    if (dir.back() == '/' || dir.back() == '\\') return dir + name;
    const char sep = dir.find('\\') != std::string::npos && dir.find('/') == std::string::npos ? '\\' : '/';
    return dir + sep + name;
}

enum class PlanningProfile {
    long_context,
    balanced,
    kv_precision,
};

PlanningProfile parse_planning_profile(const char * name) {
    if (name && std::strcmp(name, "balanced") == 0) return PlanningProfile::balanced;
    if (name && std::strcmp(name, "kv_precision") == 0) return PlanningProfile::kv_precision;
    return PlanningProfile::long_context;
}

struct PlanCandidate {
    uint32_t ctx = 0;
    const char * kv_type = nullptr;
    double kv_gib = 0;
    double total_gib = 0;
};

std::vector<PlanCandidate> enumerate_candidates(const ModelInfo & m, double weights_gib,
                                                 double state_gib, double reserve_gib,
                                                 uint32_t ctx_cap, const char * kv_force,
                                                 PlanningProfile profile) {
    static const uint32_t contexts[] = { 131072, 65536, 32768, 16384, 8192, 4096 };
    static const char * types[] = { "f16", "q8_0", "q4_0" };
    std::vector<PlanCandidate> candidates;
    const auto add = [&](uint32_t ctx, const char * type) {
        if (ctx > ctx_cap || (kv_force && std::strcmp(type, kv_force) != 0)) return;
        // The legacy profiles reserve f16 for short contexts unless the user
        // explicitly forces it. The precision profile considers it everywhere.
        if (profile != PlanningProfile::kv_precision && !kv_force &&
            std::strcmp(type, "f16") == 0 && ctx > 8192) return;
        const double kv = kv_bytes_per_token(m, type) * ctx / kGiB;
        candidates.push_back({ ctx, type, kv, weights_gib + kv + state_gib + reserve_gib });
    };
    if (profile == PlanningProfile::kv_precision) {
        for (const char * type : types) {
            for (uint32_t ctx : contexts) add(ctx, type);
        }
    } else {
        for (uint32_t ctx : contexts) {
            for (const char * type : types) add(ctx, type);
        }
    }
    return candidates;
}

struct SplitName {
    bool present = false;
    bool valid = false;
    uint32_t index = 0;
    uint32_t count = 0;
    std::string prefix;
};

SplitName split_name(const std::string & path) {
    SplitName out;
    const std::string name = basename(path);
    const std::string folded = lower(name);
    if (name.size() < 5 || folded.substr(name.size() - 5) != ".gguf") return out;
    const size_t of = folded.rfind("-of-");
    if (of == std::string::npos) return out;
    const size_t ext = name.size() - 5;
    if (of + 4 > ext) return out;

    const size_t left_separator = of > 0 ? folded.rfind('-', of - 1) : std::string::npos;
    const size_t left_begin = left_separator == std::string::npos ? 0 : left_separator + 1;
    const size_t right_begin = of + 4;
    const size_t right_end_found = folded.find('-', right_begin);
    const size_t right_end = right_end_found == std::string::npos || right_end_found > ext ? ext : right_end_found;
    const auto all_digits = [&](size_t begin, size_t end) {
        if (begin == end) return false;
        for (size_t i = begin; i < end; ++i) {
            if (!std::isdigit((unsigned char) folded[i])) return false;
        }
        return true;
    };
    const bool numeric_left = all_digits(left_begin, of);
    const bool numeric_right = all_digits(right_begin, right_end);
    // An ordinary name such as "Mixture-of-Experts-Q4.gguf" contains the
    // delimiter but has no numeric shard component. Keep it a single-file name.
    if (!numeric_left && !numeric_right) return out;

    out.present = true;
    if (of < 6 || left_separator == std::string::npos || left_separator != of - 6 ||
        of + 9 != ext || !numeric_left || !numeric_right) return out;
    for (size_t i = of - 5; i < of; ++i) {
        out.index = out.index * 10 + (uint32_t) (folded[i] - '0');
    }
    for (size_t i = of + 4; i < ext; ++i) {
        out.count = out.count * 10 + (uint32_t) (folded[i] - '0');
    }
    out.prefix = name.substr(0, left_separator);
    out.valid = !out.prefix.empty() && out.index > 0 && out.count > 0 && out.index <= out.count;
    return out;
}

bool is_unsigned_type(uint32_t t) {
    return t == U8 || t == U16 || t == U32 || t == U64;
}

bool is_integer_type(uint32_t t) {
    return is_unsigned_type(t) || t == I8 || t == I16 || t == I32 || t == I64 || t == BOOL;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (a && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

void reject(ModelInfo & m, const std::string & why) {
    m.ok = false;
    m.error = why;
}

}  // namespace

// The vision projector for a model: an "mmproj" .gguf in the same folder.
// With several, the one whose name shares the longest start with the
// model's, once "mmproj" and separators are set aside
// ("mmproj-gemma-4-E4B-f16.gguf" for "gemma-4-E4B_q4_0-it.gguf").
std::string find_mmproj(const std::string & model_path) {
    const std::string dir = dirname(model_path);
    const auto key = [](std::string n) {
        n = lower(n);
        if (const size_t at = n.find("mmproj"); at != std::string::npos) {
            n.erase(at, 6);
        }
        n.erase(std::remove_if(n.begin(), n.end(), [](char c) { return c == '-' || c == '_' || c == '.'; }), n.end());
        return n;
    };
    const std::string want = key(basename(model_path));
    std::string best;
    size_t      best_len = 0;
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return best;
    }
    while (dirent * e = readdir(d)) {
        const std::string n = e->d_name;
        if (lower(n).find("mmproj") == std::string::npos || n.size() < 6 || lower(n.substr(n.size() - 5)) != ".gguf") {
            continue;
        }
        const std::string k = key(n);
        size_t common = 0;
        while (common < k.size() && common < want.size() && k[common] == want[common]) {
            ++common;
        }
        // At least the family's name in common ("gemma4e4b"), so a lone
        // projector of another model is not paired.
        if (common >= 5 && common > best_len) {
            best = join_path(dir, n);
            best_len = common;
        }
    }
    closedir(d);
    return best;
}

namespace {

// Reads a bounded GGUF header from `f` (and closes it). `file_bytes` is the
// whole file size when `f` holds only its start.
ModelInfo parse_model_info(Reader r, const std::string & path, uint64_t file_bytes,
                           bool require_model_shape = true, uint64_t * header_bytes = nullptr) {
    FILE * f = r.f;
    ModelInfo m;
    m.path      = path;
    m.file_name = basename(path);
    m.name      = m.file_name;

    if (!f && !r.mem) {
        m.error = "cannot open";
        return m;
    }
    const uint32_t magic = r.get<uint32_t>();
    const uint32_t version = r.get<uint32_t>();
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    if (r.bad) {
        if (f) {
            std::fclose(f);
        }
        m.error = "metadata could not be read";
        return m;
    }
    if (magic != 0x46554747u) {  // "GGUF"
        if (f) std::fclose(f);
        m.error = "not a GGUF file";
        return m;
    }
    if (version < 2 || version > 3 || n_kv > kMaxMetadataItems) {
        if (f) std::fclose(f);
        m.error = "unsupported or invalid GGUF header";
        return m;
    }

    std::map<std::string, uint64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, uint32_t> types, array_types;
    std::set<std::string> keys;
    for (uint64_t i = 0; i < n_kv && !r.bad; i++) {
        const std::string key = r.str();
        const uint32_t    t   = r.get<uint32_t>();
        if (r.bad || key.empty() || !keys.insert(key).second) {
            r.bad = true;
            break;
        }
        types[key] = t;
        if (t == STRING) {
            strs[key] = r.str();
        } else if (t == ARRAY) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n  = r.get<uint64_t>();
            if (r.bad || n > kMaxArrayItems) {
                r.bad = true;
                break;
            }
            array_types[key] = et;
            if (et == STRING) {
                for (uint64_t j = 0; j < n && !r.bad; j++) {
                    const uint64_t len = r.get<uint64_t>();
                    if (r.bad || len > kMaxHeaderBytes) {
                        r.bad = true;
                        break;
                    }
                    r.skip(len);
                }
            } else if (scalar_size(et) == 0) {
                r.bad = true;
            } else if (n <= 4096) {
                uint64_t mx = 0;
                for (uint64_t j = 0; j < n; j++) {
                    mx = std::max(mx, read_int(r, et));
                }
                ints[key] = mx;
            } else {
                const uint64_t elem_size = scalar_size(et);
                if (n > UINT64_MAX / elem_size) {
                    r.bad = true;
                } else {
                    r.skip(n * elem_size);
                }
            }
        } else {
            ints[key] = read_int(r, t);
        }
    }
    m.file_bytes = file_bytes ? file_bytes : r.size;

    const std::string split_count_key = "split.count";
    const std::string split_index_key = "split.no";
    const bool has_split_count = keys.count(split_count_key) != 0;
    const bool has_split_index = keys.count(split_index_key) != 0;
    m.split_metadata_present = has_split_count || has_split_index;
    bool split_bad = false;
    if (has_split_count) {
        const auto type = types.find(split_count_key);
        const auto val = ints.find(split_count_key);
        if (type == types.end() || !is_unsigned_type(type->second) || val == ints.end()) {
            split_bad = true;
            m.split_count = 0;
        } else {
            const uint64_t count = val->second;
            m.split_count = count <= UINT32_MAX ? (uint32_t) count : UINT32_MAX;
            if (count == 0 || count > kMaxSplitCount) split_bad = true;
        }
    }
    if (has_split_index) {
        const auto type = types.find(split_index_key);
        const auto val = ints.find(split_index_key);
        if (type == types.end() || !is_unsigned_type(type->second) || val == ints.end()) {
            split_bad = true;
            m.split_index = UINT32_MAX;
        } else {
            const uint64_t index = val->second;
            m.split_index = index <= UINT32_MAX ? (uint32_t) index : UINT32_MAX;
            if (index >= kMaxSplitCount) split_bad = true;
        }
    }
    if (has_split_index && !has_split_count) split_bad = true;
    if (has_split_count && m.split_count > 1 && !has_split_index) split_bad = true;
    if (has_split_count && m.split_count <= 1 && has_split_index && m.split_index != 0) split_bad = true;
    if (has_split_count && m.split_count > 0 && m.split_index >= m.split_count) split_bad = true;

    const std::string stripped_key = "betelgeuse.stripped_experts";
    const bool has_stripped = keys.count(stripped_key) != 0;
    if (has_stripped) {
        const auto type = types.find(stripped_key);
        const auto val = ints.find(stripped_key);
        if (type == types.end() || !is_integer_type(type->second) || val == ints.end()) {
            m.stripped_experts = true;
        } else {
            m.stripped_experts = val->second != 0;
        }
    }

    if (r.bad) {
        if (f) std::fclose(f);
        m.error = "metadata could not be read";
        return m;
    }
    if (split_bad) {
        if (f) std::fclose(f);
        m.error = "invalid GGUF split metadata";
        return m;
    }
    if (has_stripped && !is_integer_type(types[stripped_key])) {
        if (f) std::fclose(f);
        m.error = "invalid stripped expert metadata";
        return m;
    }
    if (m.stripped_experts) {
        if (f) std::fclose(f);
        m.error = "unsupported stripped expert bundle";
        return m;
    }

    const auto arch_type = types.find("general.architecture");
    if ((keys.count("general.architecture") &&
         (arch_type == types.end() || arch_type->second != STRING)) ||
        (require_model_shape && (arch_type == types.end() || strs["general.architecture"].empty()))) {
        if (f) std::fclose(f);
        m.error = "missing or invalid architecture metadata";
        return m;
    }
    const auto name_type = types.find("general.name");
    if (keys.count("general.name") && (name_type == types.end() || name_type->second != STRING)) {
        if (f) std::fclose(f);
        m.error = "invalid model name metadata";
        return m;
    }
    m.arch = strs["general.architecture"];
    if (!strs["general.name"].empty()) {
        m.name = strs["general.name"];
    }
    const std::string a = m.arch + ".";
    const std::string decision_key = a + "decision.type";
    const auto decision_type = types.find(decision_key);
    if (keys.count(decision_key) && (decision_type == types.end() || decision_type->second != STRING)) {
        if (f) std::fclose(f);
        m.error = "invalid decision metadata";
        return m;
    }
    m.decision = strs[decision_key];
    bool shape_bad = false;
    auto num = [&](const std::string & k, uint64_t def = 0) {
        const std::string key = a + k;
        auto it = ints.find(key);
        if (it == ints.end()) {
            if (keys.count(key)) shape_bad = true;
            return def;
        }
        const auto type = types.find(key);
        if (type == types.end()) {
            shape_bad = true;
            return def;
        }
        const uint32_t value_type = type->second == ARRAY ? array_types[key] : type->second;
        if (value_type == BOOL || !is_integer_type(value_type)) {
            shape_bad = true;
            return def;
        }
        return it->second;
    };
    // block_count includes the MTP blocks some files carry (Qwen 3.8's
    // UD-Q2_K_XL has 65); they are not part of the main stack.
    const uint64_t blocks = num("block_count");
    const uint64_t nextn = num("nextn_predict_layers");
    if (nextn > blocks || blocks - nextn > UINT32_MAX) shape_bad = true;
    m.n_layer         = nextn <= blocks && blocks - nextn <= UINT32_MAX ? (uint32_t) (blocks - nextn) : 0;
    const uint64_t n_head = num("attention.head_count");
    const uint64_t n_head_kv = num("attention.head_count_kv", n_head);
    const uint64_t n_embd = num("embedding_length");
    const uint64_t head_dim = n_head ? n_embd / n_head : 0;
    const uint64_t key_length = num("attention.key_length", head_dim);
    const uint64_t value_length = num("attention.value_length", head_dim);
    const uint64_t full_attn_every = std::max<uint64_t>(1, num("full_attention_interval", 1));
    const uint64_t n_ctx_train = num("context_length");
    const uint64_t n_expert = num("expert_count");
    const uint64_t n_expert_used = num("expert_used_count");
    if (n_head > UINT32_MAX || n_head_kv > UINT32_MAX || key_length > UINT32_MAX ||
        value_length > UINT32_MAX || full_attn_every > UINT32_MAX || n_ctx_train > UINT32_MAX ||
        n_expert > UINT32_MAX || n_expert_used > UINT32_MAX || head_dim > UINT32_MAX) {
        shape_bad = true;
    }
    m.n_head_kv       = n_head_kv <= UINT32_MAX ? (uint32_t) n_head_kv : 0;
    m.key_length      = key_length <= UINT32_MAX ? (uint32_t) key_length : 0;
    m.value_length    = value_length <= UINT32_MAX ? (uint32_t) value_length : 0;
    m.full_attn_every = full_attn_every <= UINT32_MAX ? (uint32_t) full_attn_every : 1;
    m.n_ctx_train     = n_ctx_train <= UINT32_MAX ? (uint32_t) n_ctx_train : 0;
    m.n_expert        = n_expert <= UINT32_MAX ? (uint32_t) n_expert : 0;
    m.n_expert_used   = n_expert_used <= UINT32_MAX ? (uint32_t) n_expert_used : 0;
    if (m.n_kv_layers() > 0 && (!m.n_head_kv || !m.key_length || !m.value_length)) shape_bad = true;

    // Recurrent layers (Gated DeltaNet, Mamba): a fixed f32 state each,
    // the recurrent matrix plus the convolution window.
    const uint64_t inner = num("ssm.inner_size"), state = num("ssm.state_size");
    if ((inner == 0) != (state == 0)) shape_bad = true;
    if (inner && state) {
        const uint64_t groups = num("ssm.group_count", 1), conv = num("ssm.conv_kernel", 4);
        const uint64_t n_rec  = m.n_layer - m.n_kv_layers();
        uint64_t inner_state, groups_state, grouped_state, conv_state, per_layer, total;
        if (conv == 0 || !checked_mul(inner, state, inner_state) ||
            !checked_mul(groups, state, groups_state) || !checked_mul(groups_state, 2, groups_state) ||
            !checked_add(inner, groups_state, grouped_state) ||
            !checked_mul(conv - 1, grouped_state, conv_state) ||
            !checked_add(inner_state, conv_state, per_layer) ||
            !checked_mul(n_rec, per_layer, total) || !checked_mul(total, 4, m.state_bytes)) {
            shape_bad = true;
        }
    }
    m.ok = !shape_bad && (!require_model_shape || m.n_layer > 0);
    if (!m.ok) {
        if (f) std::fclose(f);
        m.error = shape_bad ? "invalid model shape metadata" : "no layer count for architecture '" + m.arch + "'";
        return m;
    }

    m.non_expert_bytes = m.file_bytes;

    // Tensor directory parsing (non-fatal for normal resident planning)
    if (n_tensors > 0) {
        struct ParsedTensor {
            std::string name;
            uint32_t n_dims = 0;
            uint64_t ne[4] = { 1, 1, 1, 1 };
            uint32_t type = 0;
            uint64_t offset = 0;
            uint64_t row_bytes = 0;
            uint64_t nb2 = 0;
            uint64_t nbytes = 0;
        };
        std::vector<ParsedTensor> tensors;
        tensors.reserve((size_t) std::min<uint64_t>(n_tensors, 4096));
        std::set<std::string> tensor_names;
        bool dir_ok = true;

        for (uint64_t i = 0; i < n_tensors; ++i) {
            if (r.bad || !r.can_read(8)) {
                dir_ok = false;
                break;
            }
            std::string tname = r.str();
            if (r.bad || tname.empty() || !tensor_names.insert(tname).second) {
                dir_ok = false;
                break;
            }
            const uint32_t n_dims = r.get<uint32_t>();
            if (r.bad || n_dims < 1 || n_dims > 4) {
                dir_ok = false;
                break;
            }
            uint64_t ne[4] = { 1, 1, 1, 1 };
            for (uint32_t d = 0; d < n_dims; ++d) {
                ne[d] = r.get<uint64_t>();
                if (r.bad || ne[d] == 0) {
                    dir_ok = false;
                    break;
                }
            }
            if (!dir_ok) break;
            const uint32_t type = r.get<uint32_t>();
            const uint64_t offset = r.get<uint64_t>();
            if (r.bad) {
                dir_ok = false;
                break;
            }

            GgmlQuantTraits traits;
            if (!get_ggml_traits(type, traits) || (ne[0] % traits.block_size != 0)) {
                dir_ok = false;
                break;
            }
            uint64_t row_bytes = 0, nb2 = 0, nbytes = 0;
            if (!checked_mul(ne[0] / traits.block_size, traits.type_size, row_bytes) ||
                !checked_mul(ne[1], row_bytes, nb2) ||
                !checked_mul(ne[2], ne[3], nbytes) ||
                !checked_mul(nb2, nbytes, nbytes)) {
                dir_ok = false;
                break;
            }
            tensors.push_back({ std::move(tname), n_dims, { ne[0], ne[1], ne[2], ne[3] }, type, offset, row_bytes, nb2, nbytes });
        }

        if (r.bad) {
            r.bad = false;
            dir_ok = false;
        }

        if (!dir_ok || tensors.size() != n_tensors) {
            m.moe_paging_supported = false;
            m.moe_paging_error = "GGUF tensor directory is required for expert paging";
        } else if (m.arch != "qwen3moe") {
            m.moe_paging_supported = false;
            m.moe_paging_error = "expert paging only supports qwen3moe";
        } else if (m.split_count > 1) {
            m.moe_paging_supported = false;
            m.moe_paging_error = "expert paging does not support split GGUF files";
        } else if (m.stripped_experts) {
            m.moe_paging_supported = false;
            m.moe_paging_error = "unsupported stripped expert bundle";
        } else if (m.n_expert == 0 || m.n_expert_used == 0) {
            m.moe_paging_supported = false;
            m.moe_paging_error = "expert paging requires expert metadata";
        } else {
            struct LayerExperts {
                const ParsedTensor * gate = nullptr;
                const ParsedTensor * up = nullptr;
                const ParsedTensor * down = nullptr;
            };
            std::vector<LayerExperts> layer_experts(m.n_layer);
            bool expert_valid = true;
            std::string expert_err;

            for (const auto & t : tensors) {
                if (t.name.find("_exps.") != std::string::npos) {
                    bool valid_name = false;
                    uint32_t layer_idx = 0;
                    std::string role;
                    if (t.name.rfind("blk.", 0) == 0) {
                        const size_t dot1 = 3;
                        const size_t dot2 = t.name.find('.', dot1 + 1);
                        if (dot2 != std::string::npos) {
                            const std::string lstr = t.name.substr(dot1 + 1, dot2 - (dot1 + 1));
                            bool all_digits = !lstr.empty();
                            for (char c : lstr) {
                                if (!std::isdigit((unsigned char) c)) all_digits = false;
                            }
                            if (all_digits) {
                                layer_idx = (uint32_t) std::stoul(lstr);
                                const std::string suffix = t.name.substr(dot2 + 1);
                                if (suffix == "ffn_gate_exps.weight") {
                                    role = "gate";
                                    valid_name = true;
                                } else if (suffix == "ffn_up_exps.weight") {
                                    role = "up";
                                    valid_name = true;
                                } else if (suffix == "ffn_down_exps.weight") {
                                    role = "down";
                                    valid_name = true;
                                }
                            }
                        }
                    }
                    if (!valid_name) {
                        expert_valid = false;
                        expert_err = "unrecognized expert tensor: " + t.name;
                        break;
                    }
                    if (layer_idx >= m.n_layer) {
                        expert_valid = false;
                        expert_err = "expert tensor layer index out of bounds: " + t.name;
                        break;
                    }
                    if (t.n_dims != 3 || t.ne[2] != m.n_expert) {
                        expert_valid = false;
                        expert_err = "expert tensor has unexpected dimensions: " + t.name;
                        break;
                    }
                    if (role == "gate") {
                        if (layer_experts[layer_idx].gate) {
                            expert_valid = false;
                            expert_err = "duplicate gate tensor in layer " + std::to_string(layer_idx);
                            break;
                        }
                        layer_experts[layer_idx].gate = &t;
                    } else if (role == "up") {
                        if (layer_experts[layer_idx].up) {
                            expert_valid = false;
                            expert_err = "duplicate up tensor in layer " + std::to_string(layer_idx);
                            break;
                        }
                        layer_experts[layer_idx].up = &t;
                    } else if (role == "down") {
                        if (layer_experts[layer_idx].down) {
                            expert_valid = false;
                            expert_err = "duplicate down tensor in layer " + std::to_string(layer_idx);
                            break;
                        }
                        layer_experts[layer_idx].down = &t;
                    }
                }
            }

            if (expert_valid) {
                for (uint32_t l = 0; l < m.n_layer; ++l) {
                    const auto & le = layer_experts[l];
                    if (!le.gate || !le.up || !le.down) {
                        expert_valid = false;
                        expert_err = "missing expert tensor in layer " + std::to_string(l);
                        break;
                    }
                    if (le.gate->ne[1] != le.up->ne[1] || le.gate->ne[1] != le.down->ne[0] ||
                        le.gate->ne[0] != le.down->ne[1]) {
                        expert_valid = false;
                        expert_err = "expert tensor dimension mismatch in layer " + std::to_string(l);
                        break;
                    }
                }
            }

            if (!expert_valid) {
                m.moe_paging_supported = false;
                m.moe_paging_error = expert_err;
            } else {
                uint64_t total_exp_bytes = 0;
                for (uint32_t l = 0; l < m.n_layer; ++l) {
                    total_exp_bytes += layer_experts[l].gate->nbytes;
                    total_exp_bytes += layer_experts[l].up->nbytes;
                    total_exp_bytes += layer_experts[l].down->nbytes;
                }
                m.expert_bytes = total_exp_bytes;
                m.non_expert_bytes = m.file_bytes >= total_exp_bytes ? m.file_bytes - total_exp_bytes : 0;

                struct TensorSig {
                    uint32_t type = 0;
                    uint64_t ne[3] = {};
                    uint64_t nb2 = 0;
                    bool operator<(const TensorSig & o) const {
                        return std::tie(type, ne[0], ne[1], ne[2], nb2) <
                               std::tie(o.type, o.ne[0], o.ne[1], o.ne[2], o.nb2);
                    }
                };
                struct GroupSig {
                    TensorSig gate;
                    TensorSig up;
                    TensorSig down;
                    bool operator<(const GroupSig & o) const {
                        return std::tie(gate, up, down) < std::tie(o.gate, o.up, o.down);
                    }
                };
                std::set<GroupSig> unique_groups;
                for (uint32_t l = 0; l < m.n_layer; ++l) {
                    const auto & le = layer_experts[l];
                    GroupSig sig = {
                        { le.gate->type, { le.gate->ne[0], le.gate->ne[1], le.gate->ne[2] }, le.gate->nb2 },
                        { le.up->type,   { le.up->ne[0],   le.up->ne[1],   le.up->ne[2]   }, le.up->nb2 },
                        { le.down->type, { le.down->ne[0], le.down->ne[1], le.down->ne[2] }, le.down->nb2 },
                    };
                    unique_groups.insert(sig);
                }

                const uint64_t slots_min = std::min<uint64_t>(m.n_expert_used, m.n_expert);
                uint64_t group_bytes_sum = 0;
                uint64_t max_slice_bytes = 0;
                for (const auto & g : unique_groups) {
                    const uint64_t g_min = align_up(g.gate.nb2 * (slots_min + 1), 1024ull) +
                                           align_up(g.up.nb2   * (slots_min + 1), 1024ull) +
                                           align_up(g.down.nb2 * (slots_min + 1), 1024ull);
                    const uint64_t staging = std::max({ g.gate.nb2, g.up.nb2, g.down.nb2 });
                    group_bytes_sum += g_min + staging;
                    max_slice_bytes = std::max({ max_slice_bytes, g.gate.nb2, g.up.nb2, g.down.nb2 });
                }
                const uint64_t host_map_bytes = (uint64_t) m.n_layer * align_up((uint64_t) m.n_expert * 4ull, 1024ull);
                m.moe_cache_min_bytes = group_bytes_sum + host_map_bytes;
                m.moe_max_expert_slice_bytes = max_slice_bytes;
                m.moe_paging_supported = true;
                m.moe_paging_error.clear();
            }
        }
    } else {
        m.non_expert_bytes = m.file_bytes;
        if (m.arch != "qwen3moe") {
            m.moe_paging_error = "expert paging only supports qwen3moe";
        } else if (m.split_count > 1) {
            m.moe_paging_error = "expert paging does not support split GGUF files";
        } else {
            m.moe_paging_error = "GGUF tensor directory is required for expert paging";
        }
    }

    if (f) {
        std::fclose(f);
    }
    if (header_bytes) *header_bytes = r.pos;
    return m;
}

bool get_regular_file_size(const std::string & path, uint64_t & bytes) {
    struct stat st = {};
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) return false;
    bytes = (uint64_t) st.st_size;
    return true;
}

ModelInfo parse_local_header(const std::string & path, uint64_t bytes, bool require_model_shape = true) {
    Reader r;
    r.f = std::fopen(path.c_str(), "rb");
    return parse_model_info(r, path, bytes, require_model_shape);
}

bool same_model_identity(const ModelInfo & a, const ModelInfo & b) {
    if (a.arch != b.arch || a.decision != b.decision || a.n_layer != b.n_layer ||
        a.n_head_kv != b.n_head_kv || a.key_length != b.key_length || a.value_length != b.value_length ||
        a.full_attn_every != b.full_attn_every || a.n_ctx_train != b.n_ctx_train ||
        a.n_expert != b.n_expert || a.n_expert_used != b.n_expert_used || a.state_bytes != b.state_bytes) {
        return false;
    }
    const bool a_has_name = a.name != a.file_name;
    const bool b_has_name = b.name != b.file_name;
    return a_has_name == b_has_name && (!a_has_name || a.name == b.name);
}

std::string split_shard_name(const std::string & prefix, uint32_t index, uint32_t count) {
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "-%05u-of-%05u.gguf", index + 1, count);
    return prefix + suffix;
}

}  // namespace

ModelInfo read_model_info(const std::string & path) {
    uint64_t first_bytes = 0;
    if (!get_regular_file_size(path, first_bytes)) {
        ModelInfo m;
        m.path = path;
        m.file_name = basename(path);
        m.name = m.file_name;
        m.error = "cannot read local model file";
        return m;
    }
    ModelInfo m = parse_local_header(path, first_bytes);
    m.mmproj = find_mmproj(path);
    if (!m.mmproj.empty()) {
        struct stat st = {};
        m.mmproj_bytes = stat(m.mmproj.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
    }
    if (!m.ok) return m;

    const SplitName hint = split_name(path);
    if (m.split_count == 1) {
        if (m.split_index != 0 || (hint.present && (!hint.valid || hint.index != 1 || hint.count != 1))) {
            reject(m, "model filename conflicts with GGUF split metadata");
        } else if (!m.split_metadata_present && hint.present) {
            reject(m, "split filename is missing GGUF split metadata");
        }
        return m;
    }
    if (!m.split_metadata_present || m.split_index != 0 || !hint.present || !hint.valid ||
        hint.index != 1 || hint.count != m.split_count) {
        reject(m, "split model needs its canonical first shard and matching metadata");
        return m;
    }

    const std::string dir = dirname(path);
    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < m.split_count; ++i) {
        const std::string shard_path = join_path(dir, split_shard_name(hint.prefix, i, m.split_count));
        ModelInfo shard;
        uint64_t shard_bytes = 0;
        if (i == 0) {
            shard = m;
            shard_bytes = first_bytes;
        } else {
            if (!get_regular_file_size(shard_path, shard_bytes)) {
                reject(m, "missing or invalid GGUF shard " + std::to_string(i + 1));
                return m;
            }
            shard = parse_local_header(shard_path, shard_bytes, false);
        }
        if (!shard.ok || !shard.split_metadata_present || shard.split_count != m.split_count ||
            shard.split_index != i || (!shard.arch.empty() && !same_model_identity(m, shard))) {
            reject(m, "GGUF shard metadata does not match the model split");
            return m;
        }
        if (!checked_add(total_bytes, shard_bytes, total_bytes)) {
            reject(m, "GGUF split size overflows the supported range");
            return m;
        }
    }
    m.file_bytes = total_bytes;
    m.moe_paging_supported = false;
    m.moe_paging_error = "expert paging does not support split GGUF files";
    return m;
}

ModelInfo read_model_header(const std::string & name, const std::string & head, uint64_t file_bytes) {
    Reader r;
    r.mem  = head.data();
    r.size = head.size();
    uint64_t parsed_header_bytes = 0;
    ModelInfo m = parse_model_info(r, name, file_bytes, true, &parsed_header_bytes);
    if (file_bytes == 0) {
        m.file_bytes = 0;
        if (m.ok) reject(m, "remote model file size is unknown");
        return m;
    }
    if (m.ok && (file_bytes < parsed_header_bytes || file_bytes < head.size())) {
        reject(m, "remote model size is smaller than its GGUF header");
        return m;
    }
    if (!m.ok) return m;
    const SplitName hint = split_name(name);
    if (m.split_count > 1 || (hint.present && (!hint.valid || hint.count > 1))) {
        reject(m, "remote split model cannot be validated from one header");
    } else if (hint.present && (hint.index != 1 || hint.count != m.split_count)) {
        reject(m, "model filename conflicts with GGUF split metadata");
    }
    return m;
}

double kv_bytes_per_token(const ModelInfo & m, const std::string & type) {
    const double per_elem = type == "f16" ? 2.0 : type == "q8_0" ? 34.0 / 32 : 18.0 / 32;  // q4_0
    return (double) m.n_kv_layers() * m.n_head_kv * ((double) m.key_length + (double) m.value_length) * per_elem;
}

// Sampling as each model's authors recommend it. Settings the planner works
// out (context, cache type) are not here.
const std::vector<Preset> & presets() {
    static const std::vector<Preset> table = {
        { "qwen3.8", "Qwen 3.8", "27B hybrid Gated DeltaNet; UD-Q2_K_XL holds 64k context on the GPU",
          { "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0" } },
        { "qwen3.6", "Qwen 3.6", "MoE 35B-A3B or dense 27B; the MoE decodes fastest",
          { "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0" } },
        { "qwen3.5", "Qwen 3.5", "hybrid like 3.8; 9B at Q4_K_M leaves room for long context",
          { "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0" } },
        { "ornith", "Ornith 1.5", "Qwen 3.5 based 9B",
          { "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0" } },
        { "gemma", "Gemma 4", "12B at Q3_K_M / IQ4_XS, E4B at Q4_0",
          { "--temp", "1.0", "--top-p", "0.95", "--top-k", "64" } },
        { "gpt-oss", "gpt-oss", "20B MoE, 3.6B active; MXFP4 is its only size, so context stays short",
          { "--temp", "1.0", "--top-p", "1.0" }, 16384 },
        { "granite", "Granite 4.2", "3B and 8B, fast and plain",
          { "--temp", "0.0" } },
        { "nemotron", "Nemotron 3 Nano", "hybrid Mamba, 1M context trained",
          { "--temp", "0.6", "--top-p", "0.95" } },
        { "mistral", "Mistral", "7B v0.3 and the Small line",
          { "--temp", "0.15" } },
        { "llama", "Llama 3", "8B fits at any 4-bit quant",
          { "--temp", "0.6", "--top-p", "0.9" } },
    };
    return table;
}

const Preset * find_preset(const ModelInfo & m) {
    if (!m.decision.empty()) {
        return nullptr;  // no sampling: a decision model reads its answers off one forward pass
    }
    const std::string hay = lower(m.file_name + " " + m.name + " " + m.arch);
    for (const auto & p : presets()) {
        if (hay.find(p.match) != std::string::npos) {
            return &p;
        }
    }
    return nullptr;
}

Plan plan_model(const ModelInfo & m, double budget_gib, uint32_t ctx_cap, const char * kv_force,
                const char * planning_profile, bool allow_paged_moe, uint32_t moe_cache_mib) {
    Plan p;
    if (!m.ok) {
        p.why = m.error.empty() ? "model metadata is invalid" : m.error;
        return p;
    }
    const PlanningProfile profile = parse_planning_profile(planning_profile);
    const Preset * preset = find_preset(m);
    if (preset && preset->max_ctx) {
        ctx_cap = std::min(ctx_cap, preset->max_ctx);
    }
    // Decision prompts are short; 4k keeps the cache small for the wide batch below.
    if (!m.decision.empty()) {
        ctx_cap = std::min<uint32_t>(ctx_cap, 4096);
    }
    if (m.n_ctx_train) {
        ctx_cap = std::min(ctx_cap, m.n_ctx_train);
    }
    // Balanced puts a ceiling on the context target after the model, preset,
    // and user limits have been applied.
    if (profile == PlanningProfile::balanced) {
        ctx_cap = std::min<uint32_t>(ctx_cap, 16384);
    }
    p.weights_gib = ((double) m.file_bytes + (double) m.mmproj_bytes) / kGiB;
    p.state_gib   = m.state_bytes / kGiB;
    // Compute buffers and the CPU side (the title heap shares the pool), plus
    // one prompt checkpoint of the recurrent state. Calibrated on the console:
    // Qwen3.8-27B UD-Q2_K_XL at 64k with a q4_0 cache ran with 11.38 GiB free.
    // ponytail: 0.25 GiB for the image encoder's buffers is a guess; measure on the console.
    // Decision models (laya, clef, vllm-d3, lfm2-d1-omni) and non-causal image
    // encoders (mtmd caps an image to the ubatch) need a whole prompt or image in
    // one ubatch, so they get 2048, the size #13 ran on the console.
    const bool wide = !m.decision.empty() || !m.mmproj.empty();
    // shortcut: 0.5 GiB more compute buffer for the 2048 ubatch is a guess; measure on the console.
    p.reserve_gib = 0.75 + p.state_gib + (m.mmproj.empty() ? 0.0 : 0.25) + (wide ? 0.5 : 0.0);

    const std::vector<PlanCandidate> candidates = enumerate_candidates(
        m, p.weights_gib, p.state_gib, p.reserve_gib, ctx_cap, kv_force, profile);
    const PlanCandidate * selected = nullptr;
    for (const PlanCandidate & candidate : candidates) {
        if (candidate.total_gib <= budget_gib) {
            selected = &candidate;
            break;
        }
    }

    if (selected) {
        p.fits = true;
        p.ctx = selected->ctx;
        p.kv_type = selected->kv_type;
        p.kv_gib = selected->kv_gib;
        p.total_gib = selected->total_gib;
        char why[160];
        std::snprintf(why, sizeof(why), "%s%uk context, %s cache%s: %.1f of %.1f GiB",
                      m.decision.empty() ? "" : ("decision model (" + m.decision + "), ").c_str(), p.ctx / 1024,
                      p.kv_type.c_str(), m.mmproj.empty() ? "" : ", images", p.total_gib, budget_gib);
        p.why = why;

        p.args = { "-m", m.path, "-ngl", "999", "-fit", "off", "-lm", "none",
                   "-c", std::to_string(p.ctx), "-fa", "on", "-ctk", p.kv_type, "-ctv", p.kv_type,
                   "-b", wide ? "2048" : "256", "-ub", wide ? "2048" : "64", "-np", "1",
                   // One prompt checkpoint and no RAM cache: llama-server's defaults
                   // (32 checkpoints, 8 GiB) are for a PC, and the CPU shares this pool.
                   "-ctxcp", "1", "-cram", "0",
                   "--host", "0.0.0.0", "--port", "8081" };
        if (!m.mmproj.empty()) {
            p.args.insert(p.args.end(), { "--mmproj", m.mmproj });
        }
        if (preset) {
            p.args.insert(p.args.end(), preset->args.begin(), preset->args.end());
        }
        return p;
    }

    // Model did not fit residently. Check if opt-in paged MoE can fit it.
    if (allow_paged_moe && m.n_expert > 0) {
        if (!m.moe_paging_supported || m.split_count > 1 || !m.decision.empty() || !m.mmproj.empty()) {
            p.fits = false;
            p.args.clear();
            if (!m.decision.empty()) {
                p.why = "expert paging does not support decision models";
            } else if (!m.mmproj.empty()) {
                p.why = "expert paging does not support multimodal models";
            } else if (m.split_count > 1) {
                p.why = "expert paging does not support split GGUF files";
            } else {
                p.why = m.moe_paging_error.empty() ? "expert paging is not supported" : m.moe_paging_error;
            }
            return p;
        }

        const uint64_t cache_bytes = (uint64_t) moe_cache_mib * (1024ull * 1024ull);
        if (cache_bytes < m.moe_cache_min_bytes) {
            p.fits = false;
            p.args.clear();
            const uint32_t min_mib = (uint32_t) ((m.moe_cache_min_bytes + (1024ull * 1024ull - 1)) / (1024ull * 1024ull));
            char why[160];
            std::snprintf(why, sizeof(why),
                          "paged MoE cache budget (%u MiB) is below minimum required (%u MiB)",
                          moe_cache_mib, min_mib);
            p.why = why;
            return p;
        }

        const double non_expert_weights_gib = (double) m.non_expert_bytes / kGiB;
        const double moe_cache_gib = (double) moe_cache_mib / 1024.0;
        const double staging_gib = (double) std::max<uint64_t>(m.moe_max_expert_slice_bytes * 2, 64ull << 20) / kGiB;
        const double effective_weights_gib = non_expert_weights_gib + moe_cache_gib;
        const double paged_reserve_gib = 0.75 + p.state_gib + staging_gib;

        const std::vector<PlanCandidate> paged_candidates = enumerate_candidates(
            m, effective_weights_gib, p.state_gib, paged_reserve_gib, ctx_cap, kv_force, profile);
        const PlanCandidate * paged_selected = nullptr;
        for (const PlanCandidate & candidate : paged_candidates) {
            if (candidate.total_gib <= budget_gib) {
                paged_selected = &candidate;
                break;
            }
        }

        if (!paged_selected) {
            p.fits = false;
            p.args.clear();
            p.total_gib = effective_weights_gib + p.state_gib + paged_reserve_gib;
            char why[160];
            if (!paged_candidates.empty()) {
                const PlanCandidate & minimum = *std::min_element(paged_candidates.begin(), paged_candidates.end(),
                    [](const PlanCandidate & a, const PlanCandidate & b) { return a.total_gib < b.total_gib; });
                p.total_gib = minimum.total_gib;
                std::snprintf(why, sizeof(why), "paged MoE needs %.1f GiB at %uk with %s cache; %.1f GiB is free",
                              p.total_gib, minimum.ctx / 1024, minimum.kv_type, budget_gib);
            } else {
                std::snprintf(why, sizeof(why), "no supported context/cache setting under the current limit; %.1f GiB is free",
                              budget_gib);
            }
            p.why = why;
            return p;
        }

        p.fits = true;
        p.paged_moe = true;
        p.moe_cache_mib = moe_cache_mib;
        p.moe_cache_gib = moe_cache_gib;
        p.weights_gib = non_expert_weights_gib;
        p.reserve_gib = paged_reserve_gib;
        p.ctx = paged_selected->ctx;
        p.kv_type = paged_selected->kv_type;
        p.kv_gib = paged_selected->kv_gib;
        p.total_gib = paged_selected->total_gib;

        char why[160];
        std::snprintf(why, sizeof(why), "paged MoE (%u MiB cache), %uk context, %s cache: %.1f of %.1f GiB",
                      moe_cache_mib, p.ctx / 1024, p.kv_type.c_str(), p.total_gib, budget_gib);
        p.why = why;

        p.args = { "-m", m.path, "-ngl", "999", "-fit", "off", "-lm", "none",
                   "-c", std::to_string(p.ctx), "-fa", "on", "-ctk", p.kv_type, "-ctv", p.kv_type,
                   "-b", "256", "-ub", "1", "-np", "1",
                   "--paged-moe", "--moe-cache-mib", std::to_string(moe_cache_mib),
                   "-ctxcp", "1", "-cram", "0",
                   "--host", "0.0.0.0", "--port", "8081" };
        if (preset) {
            p.args.insert(p.args.end(), preset->args.begin(), preset->args.end());
        }
        return p;
    }

    char why[160];
    p.total_gib = p.weights_gib + p.state_gib + p.reserve_gib;
    if (!candidates.empty()) {
        const PlanCandidate & minimum = *std::min_element(candidates.begin(), candidates.end(),
            [](const PlanCandidate & a, const PlanCandidate & b) { return a.total_gib < b.total_gib; });
        p.total_gib = minimum.total_gib;
        std::snprintf(why, sizeof(why), "needs %.1f GiB at %uk with %s cache; %.1f GiB is free",
                      p.total_gib, minimum.ctx / 1024, minimum.kv_type, budget_gib);
    } else {
        std::snprintf(why, sizeof(why), "no supported context/cache setting under the current limit; %.1f GiB is free",
                      budget_gib);
    }
    p.why = why;
    return p;
}

std::vector<std::string> find_models(const std::vector<std::string> & dirs) {
    std::vector<std::string> out;
    for (const auto & d : dirs) {
        DIR * dir = opendir(d.c_str());
        if (!dir) {
            continue;
        }
        while (dirent * e = readdir(dir)) {
            const std::string n = e->d_name;
            if (n.size() < 6 || lower(n.substr(n.size() - 5)) != ".gguf" || n[0] == '.') {
                continue;
            }
            // Split models: only the validated first part, which llama.cpp opens.
            // Ordinary names containing "-of-" are still single-file models.
            const SplitName split = split_name(n);
            if (split.present && (!split.valid || split.index != 1)) {
                continue;
            }
            // Vision projectors are not models on their own.
            if (lower(n).find("mmproj") != std::string::npos) {
                continue;
            }
            out.push_back(d + "/" + n);
        }
        closedir(dir);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace ps5lm
