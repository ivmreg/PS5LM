// Host check for model_plan.cpp. These compact GGUF headers are planner fixtures,
// not complete runtime models.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "model_plan.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Gguf {
    std::string kvs;
    std::string tensors_data;
    uint64_t n = 0;
    uint64_t n_tensors = 0;

    template <class T> static std::string raw(T v) { return std::string((const char *) &v, sizeof(v)); }
    static std::string str(const std::string & v) { return raw<uint64_t>(v.size()) + v; }
    Gguf & kv(const std::string & k, const std::string & v) { kvs += str(k) + raw<uint32_t>(8) + str(v); n++; return *this; }
    Gguf & kv(const std::string & k, uint32_t v) { kvs += str(k) + raw<uint32_t>(4) + raw(v); n++; return *this; }
    Gguf & kv(const std::string & k, uint64_t v) { kvs += str(k) + raw<uint32_t>(10) + raw(v); n++; return *this; }
    Gguf & kv_bool(const std::string & k, uint8_t v) { kvs += str(k) + raw<uint32_t>(7) + raw(v); n++; return *this; }
    Gguf & kv_f32(const std::string & k, float v) { kvs += str(k) + raw<uint32_t>(6) + raw(v); n++; return *this; }
    Gguf & array(const std::string & k, uint32_t elem_type, uint64_t count) {
        kvs += str(k) + raw<uint32_t>(9) + raw(elem_type) + raw(count); n++; return *this;
    }
    Gguf & tensor(const std::string & name, const std::vector<uint64_t> & dims, uint32_t type, uint64_t offset = 0) {
        tensors_data += str(name) + raw<uint32_t>((uint32_t) dims.size());
        for (uint64_t d : dims) tensors_data += raw<uint64_t>(d);
        tensors_data += raw<uint32_t>(type) + raw<uint64_t>(offset);
        n_tensors++;
        return *this;
    }
    std::string bytes() const { return "GGUF" + raw<uint32_t>(3) + raw<uint64_t>(n_tensors) + raw(n) + kvs + tensors_data; }
};

void add_qwen35(Gguf & g, bool include_kv_heads = true) {
    g.kv("general.architecture", "qwen35").kv("general.name", "Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u);
    if (include_kv_heads) g.kv("qwen35.attention.head_count_kv", 4u);
    g.kv("qwen35.embedding_length", 4096u).kv("qwen35.context_length", 262144u);
}

void add_qwen3moe_30b(Gguf & g) {
    g.kv("general.architecture", "qwen3moe").kv("general.name", "Qwen3 30B A3B")
        .kv("qwen3moe.block_count", 48u)
        .kv("qwen3moe.attention.head_count", 16u)
        .kv("qwen3moe.attention.head_count_kv", 4u)
        .kv("qwen3moe.embedding_length", 2048u)
        .kv("qwen3moe.context_length", 32768u)
        .kv("qwen3moe.expert_count", 128u)
        .kv("qwen3moe.expert_used_count", 8u)
        .kv("qwen3moe.expert_feed_forward_length", 768u);

    // Non-expert tensor ~1.00 GB (F16 [2048, 245001] = 1,003,524,096 bytes)
    g.tensor("token_embd.weight", { 2048, 245001 }, 1);

    for (uint32_t l = 0; l < 48; ++l) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        g.tensor(prefix + "ffn_gate_exps.weight", { 2048, 768, 128 }, 12);
        g.tensor(prefix + "ffn_up_exps.weight",   { 2048, 768, 128 }, 12);
        g.tensor(prefix + "ffn_down_exps.weight", { 768, 2048, 128 }, l < 24 ? 12 : 14);
    }
}

bool has_args(const ps5lm::Plan & p, const char * flag, const char * value) {
    auto it = std::find(p.args.begin(), p.args.end(), flag);
    return it != p.args.end() && it + 1 != p.args.end() && *(it + 1) == value;
}

void write_file(const std::string & path, const std::string & bytes, size_t padding = 0) {
    std::ofstream out(path, std::ios::binary);
    assert(out.good());
    out.write(bytes.data(), (std::streamsize) bytes.size());
    for (size_t i = 0; i < padding; ++i) out.put((char) ('A' + (i % 26)));
    assert(out.good());
}

uint64_t file_size(const std::string & path) {
    struct stat st = {};
    assert(stat(path.c_str(), &st) == 0 && st.st_size > 0);
    return (uint64_t) st.st_size;
}

}  // namespace

int main(int argc, char ** argv) {
    const double budget = 11.38;

    Gguf clef;
    clef.kv("general.architecture", "qwen35").kv("general.name", "Clef Flash Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u).kv("qwen35.decision.type", "clef");
    const ps5lm::ModelInfo d = ps5lm::read_model_header("clef-flash-Q4_K_M.gguf", clef.bytes(), 5ull << 30);
    assert(d.ok && d.decision == "clef");
    assert(!ps5lm::find_preset(d));
    const ps5lm::Plan dp = ps5lm::plan_model(d, budget);
    assert(dp.fits && dp.ctx == 4096 && has_args(dp, "-ub", "2048") && has_args(dp, "-b", "2048"));
    assert(dp.why.find("decision model (clef)") == 0);

    Gguf chat;
    add_qwen35(chat);
    const ps5lm::ModelInfo c = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", chat.bytes(), 5ull << 30);
    assert(c.ok && c.decision.empty() && ps5lm::find_preset(c));
    const ps5lm::Plan cp = ps5lm::plan_model(c, budget);
    assert(cp.fits && cp.ctx > 4096 && has_args(cp, "-ub", "64"));

    const std::string chat_header = chat.bytes();
    const ps5lm::ModelInfo unknown_size = ps5lm::read_model_header("unknown-size.gguf", chat_header, 0);
    assert(!unknown_size.ok && unknown_size.file_bytes == 0 && unknown_size.error.find("size is unknown") != std::string::npos);
    assert(ps5lm::plan_model(unknown_size, budget).args.empty());
    assert(ps5lm::plan_model(unknown_size, budget, 65536, nullptr, "balanced").args.empty());
    const ps5lm::ModelInfo undersized = ps5lm::read_model_header("undersized.gguf", chat_header, chat_header.size() - 1);
    assert(!undersized.ok && undersized.error.find("smaller") != std::string::npos);
    assert(ps5lm::plan_model(undersized, budget).args.empty());

    // A valid finite float in unrelated metadata is consumed without integer casting.
    Gguf ordinary_float;
    add_qwen35(ordinary_float);
    ordinary_float.kv_f32("qwen35.rope.freq_base", -100.0f);
    assert(ps5lm::read_model_header("ordinary.gguf", ordinary_float.bytes(), 1ull << 30).ok);

    Gguf string_context;
    string_context.kv("general.architecture", "qwen35").kv("qwen35.block_count", 32u)
        .kv("qwen35.attention.head_count", 16u).kv("qwen35.attention.head_count_kv", 4u)
        .kv("qwen35.embedding_length", 4096u).kv("qwen35.context_length", "4096");
    const ps5lm::ModelInfo invalid_string_context = ps5lm::read_model_header(
        "string-context.gguf", string_context.bytes(), 1ull << 30);
    assert(!invalid_string_context.ok && invalid_string_context.error.find("invalid model shape") != std::string::npos);

    const ps5lm::ModelInfo remote_mixture = ps5lm::read_model_header(
        "Mixture-of-Experts-Q4.gguf", chat_header, 1ull << 30);
    assert(remote_mixture.ok && !remote_mixture.split_metadata_present && remote_mixture.split_count == 1);
    const ps5lm::ModelInfo malformed_remote_shard = ps5lm::read_model_header(
        "Mixture-of-Experts-00001-of-0000X.gguf", chat_header, 1ull << 30);
    assert(!malformed_remote_shard.ok);

    // Split headers retain status for remote callers but cannot certify a bundle.
    Gguf remote_split;
    add_qwen35(remote_split);
    remote_split.kv("split.count", 2u).kv("split.no", 0u);
    const ps5lm::ModelInfo remote = ps5lm::read_model_header("remote-model.gguf", remote_split.bytes(), 1ull << 30);
    assert(!remote.ok && remote.split_count == 2 && remote.split_index == 0 && remote.split_metadata_present);
    assert(ps5lm::plan_model(remote, budget).args.empty());
    const ps5lm::ModelInfo unknown_split_size = ps5lm::read_model_header("remote-model.gguf", remote_split.bytes(), 0);
    assert(!unknown_split_size.ok && unknown_split_size.split_count == 2 && unknown_split_size.split_index == 0);

    // Stripped Betelgeuse bundles are rejected, including when the marker is BOOL.
    Gguf stripped;
    add_qwen35(stripped);
    stripped.kv_bool("betelgeuse.stripped_experts", 1);
    const ps5lm::ModelInfo stripped_info = ps5lm::read_model_header("stripped.gguf", stripped.bytes(), 1ull << 30);
    assert(!stripped_info.ok && stripped_info.stripped_experts);
    assert(ps5lm::plan_model(stripped_info, budget).args.empty());
    const ps5lm::ModelInfo unknown_stripped_size = ps5lm::read_model_header("stripped.gguf", stripped.bytes(), 0);
    assert(!unknown_stripped_size.ok && unknown_stripped_size.stripped_experts);

    // Truncated and malformed metadata must not produce a plan.
    const std::string valid_header = chat_header;
    const ps5lm::ModelInfo truncated = ps5lm::read_model_header("truncated.gguf", valid_header.substr(0, 11), 1ull << 30);
    assert(!truncated.ok && ps5lm::plan_model(truncated, budget).args.empty());
    assert(!ps5lm::read_model_header("truncated-metadata.gguf", valid_header.substr(0, valid_header.size() - 1), 1ull << 30).ok);

    std::string huge_kv_count = valid_header;
    huge_kv_count.replace(16, 8, Gguf::raw<uint64_t>(UINT64_MAX));
    assert(!ps5lm::read_model_header("huge-count.gguf", huge_kv_count, 1ull << 30).ok);

    Gguf huge_array;
    add_qwen35(huge_array);
    huge_array.array("tokenizer.ggml.scores", 6, UINT64_MAX);
    assert(!ps5lm::read_model_header("huge-array.gguf", huge_array.bytes(), 1ull << 30).ok);

    Gguf duplicate;
    add_qwen35(duplicate);
    duplicate.kv("general.architecture", "qwen35");
    assert(!ps5lm::read_model_header("duplicate.gguf", duplicate.bytes(), 1ull << 30).ok);

    Gguf bad_split_count;
    add_qwen35(bad_split_count);
    bad_split_count.kv("split.count", 0u).kv("split.no", 0u);
    assert(!ps5lm::read_model_header("zero-split.gguf", bad_split_count.bytes(), 1ull << 30).ok);

    Gguf missing_split_index;
    add_qwen35(missing_split_index);
    missing_split_index.kv("split.count", 2u);
    assert(!ps5lm::read_model_header("missing-index.gguf", missing_split_index.bytes(), 1ull << 30).ok);

    Gguf bad_layers;
    add_qwen35(bad_layers);
    bad_layers.kv("qwen35.nextn_predict_layers", 33u);
    assert(!ps5lm::read_model_header("bad-layers.gguf", bad_layers.bytes(), 1ull << 30).ok);

    Gguf bad_float_shape;
    add_qwen35(bad_float_shape, false);
    bad_float_shape.kv_f32("qwen35.attention.head_count_kv", -1.0f);
    assert(!ps5lm::read_model_header("bad-float-shape.gguf", bad_float_shape.bytes(), 1ull << 30).ok);

    Gguf max_dimensions;
    add_qwen35(max_dimensions);
    max_dimensions.kv("qwen35.attention.key_length", UINT32_MAX)
        .kv("qwen35.attention.value_length", UINT32_MAX);
    const ps5lm::ModelInfo max_dim_info = ps5lm::read_model_header("max-dimensions.gguf", max_dimensions.bytes(), 1ull << 30);
    assert(max_dim_info.ok && kv_bytes_per_token(max_dim_info, "f16") > 1.0e12);

    // The default minimum candidate is 4k/q4_0; failed-fit totals include its KV.
    ps5lm::ModelInfo big;
    big.ok = true;
    big.file_bytes = 20ull << 30;
    big.n_layer = 32;
    big.n_head_kv = 8;
    big.key_length = big.value_length = 128;
    const ps5lm::Plan no_fit = ps5lm::plan_model(big, budget);
    const double expected_minimum = 20.0 + kv_bytes_per_token(big, "q4_0") * 4096 / 1073741824.0 + 0.75;
    assert(!no_fit.fits && no_fit.args.empty() && std::fabs(no_fit.total_gib - expected_minimum) < 0.001);
    assert(no_fit.why.find("4k") != std::string::npos);

    // The profile changes only candidate order/context target. At this budget,
    // the default keeps 64k with q8_0, balanced targets 16k, and KV precision
    // picks f16 at the largest context that fits.
    ps5lm::ModelInfo profiles = big;
    profiles.file_bytes = 5ull << 30;
    const ps5lm::Plan legacy_default = ps5lm::plan_model(profiles, 10.0);
    const ps5lm::Plan long_context = ps5lm::plan_model(profiles, 10.0, 65536, nullptr, "long_context");
    const ps5lm::Plan balanced = ps5lm::plan_model(profiles, 10.0, 65536, nullptr, "balanced");
    const ps5lm::Plan kv_precision = ps5lm::plan_model(profiles, 10.0, 65536, nullptr, "kv_precision");
    assert(legacy_default.fits && legacy_default.ctx == 65536 && legacy_default.kv_type == "q8_0");
    assert(long_context.fits && long_context.ctx == legacy_default.ctx && long_context.kv_type == legacy_default.kv_type);
    assert(long_context.args == legacy_default.args && std::fabs(long_context.total_gib - legacy_default.total_gib) < 0.001);
    assert(balanced.fits && balanced.ctx == 16384 && balanced.kv_type == "q8_0");
    assert(kv_precision.fits && kv_precision.ctx == 32768 && kv_precision.kv_type == "f16");
    assert(std::fabs(long_context.weights_gib - balanced.weights_gib) < 0.001 &&
           std::fabs(long_context.weights_gib - kv_precision.weights_gib) < 0.001);
    const ps5lm::Plan unknown_profile = ps5lm::plan_model(profiles, 10.0, 65536, nullptr, "not-a-profile");
    assert(unknown_profile.fits && unknown_profile.ctx == legacy_default.ctx &&
           unknown_profile.kv_type == legacy_default.kv_type && unknown_profile.args == legacy_default.args);
    const ps5lm::Plan no_context_candidate = ps5lm::plan_model(profiles, 1.0, 2048);
    assert(!no_context_candidate.fits && no_context_candidate.args.empty());
    assert(std::fabs(no_context_candidate.total_gib - (5.0 + 0.75)) < 0.001);

    // Safety threshold (reserve_mib) allows squeezing longer contexts when memory is tight
    const ps5lm::Plan tight_reserve = ps5lm::plan_model(profiles, 1.0, 2048, nullptr, "long_context", false, 2048, 384);
    assert(!tight_reserve.fits && tight_reserve.args.empty());
    assert(std::fabs(tight_reserve.total_gib - (5.0 + 384.0 / 1024.0)) < 0.001);

    // Test calibrated Qwen3.8 27B UD-Q2_K_XL on console budget of 11.03 GiB (with GUI active):
    // 768 MiB default reserve caps context at 32k; 384 MiB safety threshold unlocks full 64k!
    ps5lm::ModelInfo qwen38_q2k;
    qwen38_q2k.ok = true;
    qwen38_q2k.arch = "qwen35";
    qwen38_q2k.name = "Qwen 3.8 27B";
    qwen38_q2k.file_name = "Qwen3.8-27B-UD-Q2_K_XL.gguf";
    qwen38_q2k.n_layer = 64;
    qwen38_q2k.full_attn_every = 4;
    qwen38_q2k.n_head_kv = 4;
    qwen38_q2k.key_length = 256;
    qwen38_q2k.value_length = 256;
    qwen38_q2k.file_bytes = 9828981664ull;
    qwen38_q2k.state_bytes = 157286400ull;
    const double gui_budget = 11.03;
    const ps5lm::Plan q2k_default = ps5lm::plan_model(qwen38_q2k, gui_budget, 65536, nullptr, "long_context", false, 2048, 768);
    assert(q2k_default.fits && q2k_default.ctx == 32768 && q2k_default.kv_type == "q4_0");
    const ps5lm::Plan q2k_tight = ps5lm::plan_model(qwen38_q2k, gui_budget, 65536, nullptr, "long_context", false, 2048, 384);
    assert(q2k_tight.fits && q2k_tight.ctx == 65536 && q2k_tight.kv_type == "q4_0");

    // A forced cache type and the effective user/model/preset context ceilings
    // apply for every profile. Explicitly forced f16 may exceed the automatic
    // 8k cutoff in the legacy ordering, as it did before profiles existed.
    const char * profile_names[] = { "long_context", "balanced", "kv_precision" };
    const char * forced_types[] = { "f16", "q8_0", "q4_0" };
    for (const char * profile : profile_names) {
        for (const char * forced : forced_types) {
            const ps5lm::Plan forced_plan = ps5lm::plan_model(profiles, 100.0, 65536, forced, profile);
            assert(forced_plan.fits && forced_plan.kv_type == forced);
            assert(forced_plan.ctx <= (std::string(profile) == "balanced" ? 16384u : 65536u));
        }
        const ps5lm::Plan user_cap_plan = ps5lm::plan_model(profiles, 100.0, 32768, "q8_0", profile);
        assert(user_cap_plan.fits && user_cap_plan.ctx <=
               (std::string(profile) == "balanced" ? 16384u : 32768u) && user_cap_plan.kv_type == "q8_0");
        ps5lm::ModelInfo model_capped = profiles;
        model_capped.n_ctx_train = 8192;
        const ps5lm::Plan model_cap_plan = ps5lm::plan_model(model_capped, 100.0, 65536, "q8_0", profile);
        assert(model_cap_plan.fits && model_cap_plan.ctx <= 8192 && model_cap_plan.kv_type == "q8_0");
    }
    ps5lm::ModelInfo preset_capped = profiles;
    preset_capped.file_name = "gpt-oss-test.gguf";
    preset_capped.name = preset_capped.file_name;
    for (const char * profile : profile_names) {
        const ps5lm::Plan preset_cap_plan = ps5lm::plan_model(preset_capped, 100.0, 65536, nullptr, profile);
        assert(preset_cap_plan.fits && preset_cap_plan.ctx <= 16384);
    }
    const ps5lm::Plan balanced_user_cap = ps5lm::plan_model(profiles, 100.0, 8192, "q8_0", "balanced");
    assert(balanced_user_cap.fits && balanced_user_cap.ctx == 8192 && balanced_user_cap.kv_type == "q8_0");

    ps5lm::ModelInfo no_fit_forced = big;
    const ps5lm::Plan forced_no_fit = ps5lm::plan_model(no_fit_forced, 1.0, 65536, "q8_0", "kv_precision");
    const double expected_forced_minimum = 20.0 + kv_bytes_per_token(no_fit_forced, "q8_0") * 4096 / 1073741824.0 + 0.75;
    assert(!forced_no_fit.fits && forced_no_fit.args.empty());
    assert(std::fabs(forced_no_fit.total_gib - expected_forced_minimum) < 0.001);
    assert(forced_no_fit.why.find("4k") != std::string::npos && forced_no_fit.why.find("q8_0") != std::string::npos);

    // =========================================================================
    // Opt-In Paged MoE tests (P6)
    // =========================================================================

    // 1. Synthetic Qwen3-30B-A3B-Q4_K_M.gguf
    Gguf qwen3moe;
    add_qwen3moe_30b(qwen3moe);
    const uint64_t qwen3moe_size = 18556686336ull;
    const ps5lm::ModelInfo qm = ps5lm::read_model_header("Qwen3-30B-A3B-Q4_K_M.gguf", qwen3moe.bytes(), qwen3moe_size);
    assert(qm.ok && qm.moe_paging_supported);
    assert(qm.expert_bytes == 17553162240ull);
    assert(qm.non_expert_bytes == 1003524096ull);
    assert(qm.moe_max_expert_slice_bytes == 1290240ull);
    assert(qm.moe_cache_min_bytes == 68874240ull); // ~65.68 MiB

    // Resident mode (allow_paged_moe = false) on budget = 11.38 GiB: does NOT fit
    const ps5lm::Plan res_moe_plan = ps5lm::plan_model(qm, budget);
    assert(!res_moe_plan.fits && !res_moe_plan.paged_moe);

    // Paged mode (allow_paged_moe = true, moe_cache_mib = 2048) on budget = 11.38 GiB: fits!
    const ps5lm::Plan paged_plan = ps5lm::plan_model(qm, budget, 65536, nullptr, "long_context", true, 2048);
    assert(paged_plan.fits && paged_plan.paged_moe);
    assert(paged_plan.moe_cache_mib == 2048);
    assert(std::fabs(paged_plan.moe_cache_gib - 2.0) < 0.001);
    assert(std::fabs(paged_plan.weights_gib - (double) qm.non_expert_bytes / 1073741824.0) < 0.001);
    assert(std::fabs(paged_plan.total_gib - (paged_plan.weights_gib + paged_plan.moe_cache_gib +
           paged_plan.kv_gib + paged_plan.state_gib + paged_plan.reserve_gib)) < 0.001);
    assert(paged_plan.total_gib <= budget);
    assert(has_args(paged_plan, "-ub", "1"));
    assert(has_args(paged_plan, "-np", "1"));
    assert(has_args(paged_plan, "--moe-cache-mib", "2048"));
    assert(std::find(paged_plan.args.begin(), paged_plan.args.end(), "--paged-moe") != paged_plan.args.end());
    assert(paged_plan.why.find("paged MoE (2048 MiB cache)") == 0);

    // A resident fit stays resident even when paging is enabled and the cache request is too small.
    ps5lm::ModelInfo resident_moe = qm;
    resident_moe.file_bytes = 1ull << 30;
    resident_moe.non_expert_bytes = 1ull << 29;
    resident_moe.expert_bytes = 1ull << 29;
    resident_moe.n_ctx_train = 8192;
    const ps5lm::Plan resident_only = ps5lm::plan_model(resident_moe, budget, 65536);
    const ps5lm::Plan resident_with_paging = ps5lm::plan_model(resident_moe, budget, 65536, nullptr,
                                                               "long_context", true, 16);
    assert(resident_only.fits && !resident_only.paged_moe);
    assert(resident_with_paging.fits && !resident_with_paging.paged_moe);
    assert(resident_with_paging.args == resident_only.args && resident_with_paging.ctx == resident_only.ctx &&
           resident_with_paging.kv_type == resident_only.kv_type &&
           std::fabs(resident_with_paging.total_gib - resident_only.total_gib) < 0.001);

    // Paged mode preserves profile ordering and the effective user/model context caps.
    const ps5lm::Plan paged_long_context = ps5lm::plan_model(qm, budget, 65536, nullptr, "long_context", true, 2048);
    const ps5lm::Plan paged_balanced = ps5lm::plan_model(qm, budget, 65536, nullptr, "balanced", true, 2048);
    const ps5lm::Plan paged_precision = ps5lm::plan_model(qm, budget, 65536, nullptr, "kv_precision", true, 2048);
    assert(paged_long_context.fits && paged_long_context.ctx == 32768 && paged_long_context.kv_type == "q8_0");
    assert(paged_balanced.fits && paged_balanced.ctx == 16384 && paged_balanced.kv_type == "q8_0");
    assert(paged_precision.fits && paged_precision.ctx == 32768 && paged_precision.kv_type == "f16");

    const ps5lm::Plan paged_user_cap = ps5lm::plan_model(qm, budget, 8192, "q8_0", "long_context", true, 2048);
    assert(paged_user_cap.fits && paged_user_cap.ctx == 8192 && paged_user_cap.kv_type == "q8_0");

    // The explicit GPU cache budget remains exact when changed; it is not traded for context.
    const ps5lm::Plan larger_cache_plan = ps5lm::plan_model(qm, budget, 65536, nullptr, "long_context", true, 3072);
    assert(larger_cache_plan.fits && larger_cache_plan.moe_cache_mib == 3072);
    assert(has_args(larger_cache_plan, "--moe-cache-mib", "3072"));
    assert(larger_cache_plan.ctx == paged_long_context.ctx && larger_cache_plan.kv_type == paged_long_context.kv_type);
    assert(std::fabs(larger_cache_plan.total_gib - paged_long_context.total_gib - 1.0) < 0.001);

    // Insufficient cache budget (moe_cache_mib = 16 MiB < 66 MiB minimum): fits == false
    const ps5lm::Plan low_cache_plan = ps5lm::plan_model(qm, budget, 65536, nullptr, "long_context", true, 16);
    assert(!low_cache_plan.fits);
    assert(low_cache_plan.why.find("minimum required") != std::string::npos);

    // A valid cache cannot make a plan fit when even the smallest context exceeds free memory.
    const ps5lm::Plan insufficient_memory = ps5lm::plan_model(qm, 3.0, 4096, "q4_0", "long_context", true, 2048);
    assert(!insufficient_memory.fits && insufficient_memory.args.empty());
    assert(insufficient_memory.total_gib > 3.0);

    // 2. Split GGUF rejection in paged mode
    Gguf split_moe_g;
    add_qwen3moe_30b(split_moe_g);
    split_moe_g.kv("split.count", 2u).kv("split.no", 0u);
    const ps5lm::ModelInfo split_moe = ps5lm::read_model_header("split-moe.gguf", split_moe_g.bytes(), qwen3moe_size);
    assert(!split_moe.ok || !split_moe.moe_paging_supported);
    const ps5lm::Plan split_paged_plan = ps5lm::plan_model(split_moe, budget, 65536, nullptr, "long_context", true, 2048);
    assert(!split_paged_plan.fits && split_paged_plan.args.empty());

    // 3. Stripped bundle rejection in both resident and paged modes
    Gguf stripped_moe_g;
    add_qwen3moe_30b(stripped_moe_g);
    stripped_moe_g.kv_bool("betelgeuse.stripped_experts", 1);
    const ps5lm::ModelInfo stripped_moe = ps5lm::read_model_header("stripped-moe.gguf", stripped_moe_g.bytes(), qwen3moe_size);
    assert(!stripped_moe.ok && stripped_moe.stripped_experts);
    const ps5lm::Plan stripped_res = ps5lm::plan_model(stripped_moe, budget, 65536, nullptr, "long_context", false, 2048);
    assert(!stripped_res.fits && stripped_res.args.empty());
    const ps5lm::Plan stripped_paged = ps5lm::plan_model(stripped_moe, budget, 65536, nullptr, "long_context", true, 2048);
    assert(!stripped_paged.fits && stripped_paged.args.empty());

    // 4. Local fixture tests/fixtures/moe_paging/qwen3moe-tiny-f32.gguf
    const ps5lm::ModelInfo tiny = ps5lm::read_model_info("tests/fixtures/moe_paging/qwen3moe-tiny-f32.gguf");
    assert(tiny.ok);
    assert(tiny.arch == "qwen3moe");
    assert(tiny.moe_paging_supported);
    assert(tiny.expert_bytes == 49152);
    assert(tiny.file_bytes == 156640);
    assert(tiny.non_expert_bytes == 107488);
    assert(tiny.moe_cache_min_bytes == 36864);
    assert(tiny.moe_max_expert_slice_bytes == 2048);

    // =========================================================================

    // The loader stores full model metadata in shard 1; later shards contain
    // only their split identity. These files test planning-size aggregation.
    char temp_template[] = "/tmp/ps5lm-model-plan-XXXXXX";
    char * temp = mkdtemp(temp_template);
    assert(temp != nullptr);
    const std::string dir = temp;
    const std::string single_path = dir + "/single.gguf";
    const std::string mixture_path = dir + "/Mixture-of-Experts-Q4.gguf";
    const std::string malformed_shard_path = dir + "/fixture-00001-of-0000X.gguf";
    const std::string first_path = dir + "/fixture-00001-of-00002.gguf";
    const std::string second_path = dir + "/fixture-00002-of-00002.gguf";
    Gguf single;
    add_qwen35(single);
    write_file(single_path, single.bytes(), 2);
    const ps5lm::ModelInfo local_single = ps5lm::read_model_info(single_path);
    assert(local_single.ok && !local_single.split_metadata_present && local_single.file_bytes == file_size(single_path));
    write_file(mixture_path, single.bytes(), 4);
    const ps5lm::ModelInfo local_mixture = ps5lm::read_model_info(mixture_path);
    assert(local_mixture.ok && !local_mixture.split_metadata_present && local_mixture.split_count == 1);
    write_file(malformed_shard_path, single.bytes());
    assert(!ps5lm::read_model_info(malformed_shard_path).ok);

    Gguf first;
    add_qwen35(first);
    first.kv("split.count", 2u).kv("split.no", 0u);
    Gguf second;
    second.kv("split.count", 2u).kv("split.no", 1u);
    write_file(first_path, first.bytes(), 3);
    write_file(second_path, second.bytes(), 7);
    const uint64_t split_sum = file_size(first_path) + file_size(second_path);
    const ps5lm::ModelInfo local_split = ps5lm::read_model_info(first_path);
    assert(local_split.ok && local_split.split_count == 2 && local_split.split_index == 0);
    assert(local_split.file_bytes == split_sum);

    std::remove(second_path.c_str());
    assert(!ps5lm::read_model_info(first_path).ok);
    write_file(second_path, second.bytes());

    Gguf wrong_index;
    wrong_index.kv("split.count", 2u).kv("split.no", 0u);
    write_file(second_path, wrong_index.bytes());
    assert(!ps5lm::read_model_info(first_path).ok);

    Gguf wrong_identity;
    wrong_identity.kv("split.count", 2u).kv("split.no", 1u).kv("general.architecture", "other");
    write_file(second_path, wrong_identity.bytes());
    assert(!ps5lm::read_model_info(first_path).ok);
    write_file(second_path, second.bytes(), 7);

    const std::vector<std::string> models = ps5lm::find_models({ dir });
    assert(models.size() == 3);
    assert(std::find(models.begin(), models.end(), mixture_path) != models.end());
    assert(std::find(models.begin(), models.end(), first_path) != models.end());
    assert(std::find(models.begin(), models.end(), single_path) != models.end());
    std::remove(single_path.c_str());
    std::remove(mixture_path.c_str());
    std::remove(malformed_shard_path.c_str());
    std::remove(first_path.c_str());
    std::remove(second_path.c_str());
    rmdir(dir.c_str());

    if (argc > 1) {
        bool saw_q2k = false, saw_iq2xxs = false;
        for (const auto & path : ps5lm::find_models({ argv[1] })) {
            const ps5lm::ModelInfo m = ps5lm::read_model_info(path);
            assert(m.ok);
            const ps5lm::Plan p = ps5lm::plan_model(m, budget);
            const ps5lm::Preset * pre = ps5lm::find_preset(m);
            std::printf("%-34s %s L%u kvL%u kvh%u k%u v%u state %.0f MiB | %s | %s\n", m.file_name.c_str(),
                        m.arch.c_str(), m.n_layer, m.n_kv_layers(), m.n_head_kv, m.key_length, m.value_length,
                        m.state_bytes / 1048576.0, p.why.c_str(), pre ? pre->label : "-");
            if (m.arch == "qwen35") {
                assert(m.n_layer == 64 && m.n_kv_layers() == 16 && m.n_head_kv == 4 && m.key_length == 256);
                assert(kv_bytes_per_token(m, "f16") == 65536.0);
                assert(m.state_bytes > 140.0 * 1048576 && m.state_bytes < 170.0 * 1048576);
                assert(pre && std::string(pre->label) == "Qwen 3.8");
            }
            if (m.file_name.find("UD-Q2_K_XL") != std::string::npos) {
                assert(p.fits && p.ctx == 65536 && p.kv_type == "q4_0");
                saw_q2k = true;
            }
            if (m.file_name.find("UD-IQ2_XXS") != std::string::npos) {
                assert(p.fits && p.ctx == 65536 && p.kv_type == "q8_0");
                saw_iq2xxs = true;
            }
        }
        assert(saw_q2k && saw_iq2xxs);
    }

    std::puts("model_plan_test: ok (synthetic headers, paged MoE planning and local split fixtures)");
    return 0;
}
