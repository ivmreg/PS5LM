// What the user can change on the Settings page, kept in
// /data/PS5LM/settings.json. Unknown or missing keys keep their defaults, so
// an old file still loads after a new setting is added.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>

namespace ps5lm {

struct Settings {
    bool        auto_load = false;  // load default_model at launch
    std::string default_model;      // a path from the library, or empty
    uint32_t    ctx_cap   = 65536;  // the planner's longest context
    std::string kv_type   = "auto"; // "auto", "f16", "q8_0" or "q4_0"
    std::string planning_profile = "long_context"; // "long_context", "balanced" or "kv_precision"
    uint32_t    reserve_mib = 768;  // GPU compute buffer safety headroom in MiB (default 768 MiB = 0.75 GiB)
    bool        sounds    = true;
    bool        tools     = true;   // file tools in the browser chat, inside the scratch folder
    bool        introspect = false;  // POST /v1/introspect: raw logits and activations (off by default)
    bool        paged_moe = false;   // opt-in SSD expert paging for beyond-RAM Qwen3-MoE models
    uint32_t    moe_cache_mib = 2048; // GPU expert slot cache budget in MiB when paged_moe is enabled
    std::string speculative = "off"; // "off", "auto", "draft"
    std::string draft_model;         // companion draft model file path
    uint32_t    draft_max = 8;       // tokens drafted per step
    uint32_t    scratch_limit_gib = 2;  // warn when the scratch folder grows past this
    std::string download_dir = "/data/PS5LM/models";  // where the market saves models

    bool operator==(const Settings &) const = default;
};

Settings load_settings(const char * path);
bool     save_settings(const char * path, const Settings & s);

}  // namespace ps5lm
