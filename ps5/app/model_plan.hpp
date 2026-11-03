// Model plans for the PS5LM app: what is in a GGUF, and the llama-server
// arguments that fit it on the console's GPU.
//
// A plan picks the longest context and the best KV cache type that fit the
// device memory left after the weights, from the model's own metadata (only
// attention layers keep a KV cache; hybrid models such as Qwen 3.5/3.8 keep
// one in four, and their recurrent layers a fixed state). Known models add
// their recommended sampling from the preset table.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ps5lm {

struct ModelInfo {
    std::string path;
    std::string file_name;
    std::string name;   // general.name, or the file name
    std::string arch;   // general.architecture
    std::string decision;  // "<arch>.decision.type" (openjev, clef, lfm2-d1, ...): answers /v1/systemone, not chat
    uint64_t    file_bytes      = 0;
    uint32_t    split_count     = 1;  // GGUF split.count, or 1 for a single file
    uint32_t    split_index     = 0;  // GGUF split.no, zero-based
    bool        split_metadata_present = false;
    bool        stripped_experts = false;
    uint32_t    n_layer         = 0;
    uint32_t    n_head_kv       = 0;
    uint32_t    key_length      = 0;   // per KV head
    uint32_t    value_length    = 0;
    uint32_t    full_attn_every = 1;   // 1: every layer is attention
    uint32_t    n_ctx_train     = 0;
    uint32_t    n_expert        = 0;   // 0: dense
    uint32_t    n_expert_used   = 0;
    uint32_t    n_mtp_layers    = 0;   // <arch>.nextn_predict_layers (e.g. Qwen 3.8 MTP / NextN)
    uint64_t    mtp_bytes       = 0;   // tensor bytes if parsed, else estimate
    std::string mtp_sidecar;           // MTP sidecar file beside the model, or empty
    std::string tokenizer_model;       // tokenizer.ggml.model (e.g. "llama", "gpt2", "qwen2")
    uint32_t    vocab_size      = 0;   // vocab token count
    uint64_t    expert_bytes    = 0;   // total byte size of routed expert weight tensors (when tensor directory is parsed)
    uint64_t    non_expert_bytes = 0;  // file_bytes minus expert_bytes (or non-expert tensor bytes + header overhead)
    uint64_t    moe_cache_min_bytes = 0; // minimum MoE cache + group staging + slot maps for 1-token paged execution
    uint64_t    moe_max_expert_slice_bytes = 0; // largest single expert slice stride (nb[2]) for staging buffer accounting
    bool        moe_paging_supported = false;
    std::string moe_paging_error;
    uint64_t    state_bytes     = 0;   // recurrent state, all layers (f32)
    std::string mmproj;                // a vision projector beside the model, or empty
    uint64_t    mmproj_bytes    = 0;
    bool        ok              = false;
    std::string error;

    uint32_t n_kv_layers() const { return full_attn_every > 1 ? n_layer / full_attn_every : n_layer; }
};

// Reads the metadata of a GGUF file (not its tensors).
// The same from the first bytes of a GGUF file (a range request to a model
// host), with the whole file's size: enough to plan before downloading.
// Fetch enough to hold the metadata (16 MiB covers large vocabularies).
ModelInfo read_model_header(const std::string & name, const std::string & head, uint64_t file_bytes);

// The vision projector ("mmproj") beside a model, or empty.
std::string find_mmproj(const std::string & model_path);

// The MTP head sidecar ("mtp") beside a model, or empty.
std::string find_mtp(const std::string & model_path);

ModelInfo read_model_info(const std::string & path);

// Bytes of KV cache per token of context, for a cache type ("f16", "q8_0", "q4_0").
double kv_bytes_per_token(const ModelInfo & m, const std::string & type);

struct Preset {
    const char * match;      // case-insensitive substring of the file or model name
    const char * label;      // what the UI calls it
    const char * note;       // one line about it on a PS5
    std::vector<std::string> args;   // sampling and template arguments
    uint32_t     max_ctx = 0;        // 0: the planner decides
};

const std::vector<Preset> & presets();
const Preset * find_preset(const ModelInfo & m);

struct SpecOptions {
    std::string mode = "off";        // "off", "mtp", "draft"
    std::string draft_path;          // for "draft" or sidecar for "mtp"
    uint64_t    draft_bytes = 0;
    uint32_t    draft_max = 8;       // tokens drafted per step
    uint32_t    draft_min = 1;
};

struct Plan {
    bool        fits = false;
    bool        paged_moe = false;
    uint32_t    moe_cache_mib = 0;
    double      moe_cache_gib = 0;
    std::string spec_mode = "off";   // "off", "mtp", "draft"
    double      spec_gib = 0;        // extra memory budget allocated for speculative decoding
    uint32_t    ctx  = 0;
    std::string kv_type;      // cache type for K and V
    double      weights_gib  = 0;
    double      kv_gib       = 0;
    double      state_gib    = 0;
    double      reserve_gib  = 0;   // compute buffers and the CPU side
    double      total_gib    = 0;
    std::string why;          // one line for the UI
    std::vector<std::string> args;  // full llama-server arguments
};

// budget_gib: device memory free before loading (the app reads it from ggml).
// kv_force: one cache type ("f16", "q8_0", "q4_0") instead of the best that fits.
// planning_profile: "long_context" (default), "balanced" or "kv_precision".
// Existing calls retain the long-context policy.
// A resident plan is preferred when it fits. In paged MoE mode, moe_cache_mib
// is a fixed requested budget; the planner does not shrink it to make a plan fit.
Plan plan_model(const ModelInfo & m, double budget_gib, uint32_t ctx_cap = 65536,
                const char * kv_force = nullptr, const char * planning_profile = "long_context",
                bool allow_paged_moe = false, uint32_t moe_cache_mib = 2048,
                uint32_t reserve_mib = 768,
                const SpecOptions * spec = nullptr);

// Every .gguf in the given folders (not recursive), sorted by name. Split
// files are listed by their first part only.
std::vector<std::string> find_models(const std::vector<std::string> & dirs);

}  // namespace ps5lm
