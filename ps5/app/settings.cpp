// SPDX-License-Identifier: GPL-3.0-or-later
#include "settings.hpp"

#include <nlohmann/json.hpp>

#include <fstream>

namespace ps5lm {

Settings load_settings(const char * path) {
    Settings      s;
    std::ifstream in(path);
    if (!in) {
        return s;
    }
    const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (!j.is_object()) {
        return s;
    }
    s.auto_load     = j.value("auto_load", s.auto_load);
    s.default_model = j.value("default_model", s.default_model);
    s.ctx_cap       = j.value("ctx_cap", s.ctx_cap);
    const auto kv = j.find("kv_type");
    if (kv != j.end() && kv->is_string()) {
        const std::string value = kv->get<std::string>();
        if (value == "auto" || value == "f16" || value == "q8_0" || value == "q4_0") {
            s.kv_type = value;
        }
    }
    const auto profile = j.find("planning_profile");
    if (profile != j.end() && profile->is_string()) {
        const std::string value = profile->get<std::string>();
        if (value == "long_context" || value == "balanced" || value == "kv_precision") {
            s.planning_profile = value;
        }
    }
    const auto reserve = j.find("reserve_mib");
    if (reserve != j.end() && reserve->is_number_unsigned()) {
        const uint64_t val = reserve->get<uint64_t>();
        if (val >= 128 && val <= 4096) {
            s.reserve_mib = (uint32_t) val;
        }
    }
    s.sounds        = j.value("sounds", s.sounds);
    s.tools         = j.value("tools", s.tools);
    s.introspect    = j.value("introspect", s.introspect);
    const auto paged = j.find("paged_moe");
    if (paged != j.end() && paged->is_boolean()) {
        s.paged_moe = paged->get<bool>();
    }
    const auto cache = j.find("moe_cache_mib");
    if (cache != j.end() && cache->is_number_unsigned()) {
        const uint64_t val = cache->get<uint64_t>();
        if (val >= 128 && val <= 8192) {
            s.moe_cache_mib = (uint32_t) val;
        }
    }
    const auto spec = j.find("speculative");
    if (spec != j.end() && spec->is_string()) {
        const std::string val = spec->get<std::string>();
        if (val == "off" || val == "auto" || val == "draft") {
            s.speculative = val;
        }
    }
    s.draft_model = j.value("draft_model", s.draft_model);
    const auto dmax = j.find("draft_max");
    if (dmax != j.end() && dmax->is_number_unsigned()) {
        const uint64_t val = dmax->get<uint64_t>();
        if (val >= 1 && val <= 64) {
            s.draft_max = (uint32_t) val;
        }
    }
    s.scratch_limit_gib = j.value("scratch_limit_gib", s.scratch_limit_gib);
    s.download_dir  = j.value("download_dir", s.download_dir);
    return s;
}

bool save_settings(const char * path, const Settings & s) {
    const nlohmann::json j = {
        { "auto_load", s.auto_load }, { "default_model", s.default_model }, { "ctx_cap", s.ctx_cap },
        { "kv_type", s.kv_type },     { "planning_profile", s.planning_profile },
        { "reserve_mib", s.reserve_mib },
        { "sounds", s.sounds },       { "tools", s.tools },
        { "introspect", s.introspect },
        { "paged_moe", s.paged_moe }, { "moe_cache_mib", s.moe_cache_mib },
        { "speculative", s.speculative }, { "draft_model", s.draft_model }, { "draft_max", s.draft_max },
        { "scratch_limit_gib", s.scratch_limit_gib }, { "download_dir", s.download_dir },
    };
    std::ofstream out(path);
    out << j.dump(2) << "\n";
    return (bool) out;
}

}  // namespace ps5lm
