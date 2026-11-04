// Host check for settings persistence and safe planning-profile validation.
//   c++ -std=c++20 -Ithird_party/llama.cpp/vendor ps5/app/settings_test.cpp ps5/app/settings.cpp -o build/settings_test
// SPDX-License-Identifier: GPL-3.0-or-later
#include "settings.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <stdlib.h>
#include <string>
#include <unistd.h>

namespace {

void write_text(const std::string & path, const std::string & contents) {
    std::ofstream out(path);
    assert(out.good());
    out << contents;
    assert(out.good());
}

}  // namespace

int main() {
    char temp_template[] = "/tmp/ps5lm-settings-XXXXXX";
    const int fd = mkstemp(temp_template);
    assert(fd >= 0);
    close(fd);
    const std::string path = temp_template;

    const ps5lm::Settings defaults = ps5lm::load_settings(path.c_str());
    assert(defaults.planning_profile == "long_context");
    assert(defaults.paged_moe == false);
    assert(defaults.moe_cache_mib == 2048);
    assert(defaults.reserve_mib == 768);
    assert(defaults.speculative == "off");
    assert(defaults.draft_model.empty());
    assert(defaults.draft_max == 8);
    write_text(path, "{}\n");
    const ps5lm::Settings empty_loaded = ps5lm::load_settings(path.c_str());
    assert(empty_loaded.planning_profile == "long_context");
    assert(empty_loaded.paged_moe == false);
    assert(empty_loaded.moe_cache_mib == 2048);
    assert(empty_loaded.reserve_mib == 768);
    assert(empty_loaded.speculative == "off");
    assert(empty_loaded.draft_model.empty());
    assert(empty_loaded.draft_max == 8);

    for (const char * name : { "long_context", "balanced", "kv_precision" }) {
        ps5lm::Settings expected;
        expected.planning_profile = name;
        expected.default_model = "/data/PS5LM/models/example.gguf";
        expected.ctx_cap = 32768;
        expected.paged_moe = true;
        expected.moe_cache_mib = 4096;
        expected.reserve_mib = 384;
        expected.speculative = "auto";
        expected.draft_model = "/data/PS5LM/models/draft.gguf";
        expected.draft_max = 12;
        assert(ps5lm::save_settings(path.c_str(), expected));
        const ps5lm::Settings loaded = ps5lm::load_settings(path.c_str());
        assert(loaded == expected);
        assert(loaded.paged_moe == true);
        assert(loaded.moe_cache_mib == 4096);
        assert(loaded.reserve_mib == 384);
        assert(loaded.speculative == "auto");
        assert(loaded.draft_model == "/data/PS5LM/models/draft.gguf");
        assert(loaded.draft_max == 12);
    }

    write_text(path, "{\"planning_profile\":\"future-policy\"}\n");
    assert(ps5lm::load_settings(path.c_str()).planning_profile == "long_context");
    write_text(path, "{\"planning_profile\":42}\n");
    assert(ps5lm::load_settings(path.c_str()).planning_profile == "long_context");
    write_text(path, "{\"planning_profile\":null}\n");
    assert(ps5lm::load_settings(path.c_str()).planning_profile == "long_context");

    // Invalid kv_type falls back to auto instead of making every plan fail.
    write_text(path, "{\"kv_type\":\"bogus\"}\n");
    assert(ps5lm::load_settings(path.c_str()).kv_type == "auto");
    write_text(path, "{\"kv_type\":42}\n");
    assert(ps5lm::load_settings(path.c_str()).kv_type == "auto");
    write_text(path, "{\"kv_type\":\"q4_0\"}\n");
    assert(ps5lm::load_settings(path.c_str()).kv_type == "q4_0");

    // Malformed paged_moe tests
    write_text(path, "{\"paged_moe\":1}\n");
    assert(ps5lm::load_settings(path.c_str()).paged_moe == false);
    write_text(path, "{\"paged_moe\":\"true\"}\n");
    assert(ps5lm::load_settings(path.c_str()).paged_moe == false);
    write_text(path, "{\"paged_moe\":null}\n");
    assert(ps5lm::load_settings(path.c_str()).paged_moe == false);

    // Malformed and out-of-range moe_cache_mib tests
    write_text(path, "{\"moe_cache_mib\":0}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);
    write_text(path, "{\"moe_cache_mib\":64}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);
    write_text(path, "{\"moe_cache_mib\":16384}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);
    write_text(path, "{\"moe_cache_mib\":-1024}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);
    write_text(path, "{\"moe_cache_mib\":\"2048\"}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);
    write_text(path, "{\"moe_cache_mib\":null}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 2048);

    // Valid range boundary checks for moe_cache_mib
    write_text(path, "{\"moe_cache_mib\":128}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 128);
    write_text(path, "{\"moe_cache_mib\":8192}\n");
    assert(ps5lm::load_settings(path.c_str()).moe_cache_mib == 8192);

    // Malformed and out-of-range reserve_mib tests
    write_text(path, "{\"reserve_mib\":0}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 768);
    write_text(path, "{\"reserve_mib\":64}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 768);
    write_text(path, "{\"reserve_mib\":8192}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 768);
    write_text(path, "{\"reserve_mib\":\"384\"}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 768);
    write_text(path, "{\"reserve_mib\":null}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 768);

    // Valid range boundary checks for reserve_mib
    write_text(path, "{\"reserve_mib\":128}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 128);
    write_text(path, "{\"reserve_mib\":4096}\n");
    assert(ps5lm::load_settings(path.c_str()).reserve_mib == 4096);

    // Malformed and invalid speculative tests
    write_text(path, "{\"speculative\":\"invalid-mode\"}\n");
    assert(ps5lm::load_settings(path.c_str()).speculative == "off");
    write_text(path, "{\"speculative\":123}\n");
    assert(ps5lm::load_settings(path.c_str()).speculative == "off");
    write_text(path, "{\"speculative\":null}\n");
    assert(ps5lm::load_settings(path.c_str()).speculative == "off");

    // Malformed and out-of-bounds draft_max tests
    write_text(path, "{\"draft_max\":0}\n");
    assert(ps5lm::load_settings(path.c_str()).draft_max == 8);
    write_text(path, "{\"draft_max\":128}\n");
    assert(ps5lm::load_settings(path.c_str()).draft_max == 8);
    write_text(path, "{\"draft_max\":\"16\"}\n");
    assert(ps5lm::load_settings(path.c_str()).draft_max == 8);
    write_text(path, "{\"draft_max\":16}\n");
    assert(ps5lm::load_settings(path.c_str()).draft_max == 16);

    std::remove(path.c_str());
    std::puts("settings_test: ok (defaults, validation and profile round trips)");
    return 0;
}
