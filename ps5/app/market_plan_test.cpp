// Host integration check for the market's remote header path:
//   gcc -std=c99 -O1 -Ithird_party/llama.cpp/vendor/hash -c \
//     third_party/llama.cpp/vendor/hash/sha256/sha256.c -o build/market-test-sha256.o
//   c++ -std=c++17 -O1 -ffunction-sections -fdata-sections -DPS5LM_MARKET_TEST_QUEUE \
//     -Ithird_party/llama.cpp/vendor -Ithird_party/llama.cpp/vendor/hash -Ips5/app \
//     ps5/app/market_plan_test.cpp ps5/app/market.cpp ps5/app/model_plan.cpp \
//     build/market-test-sha256.o -Wl,--gc-sections -pthread -o build/market_plan_test
//   build/market_plan_test
// HTTP is mocked; no request reaches the network. Queue and download output
// stay under build/.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "market.hpp"

#include "fetch.hpp"
#include "ftp_writer.hpp"
#include "model_plan.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>

namespace {

struct Gguf {
    std::string metadata;
    uint64_t count = 0;

    template <class T> static std::string raw(T value) { return std::string((const char *) &value, sizeof(value)); }
    static std::string str(const std::string & value) { return raw<uint64_t>(value.size()) + value; }

    Gguf & kv(const std::string & key, const std::string & value) {
        metadata += str(key) + raw<uint32_t>(8) + str(value);
        ++count;
        return *this;
    }
    Gguf & kv(const std::string & key, uint32_t value) {
        metadata += str(key) + raw<uint32_t>(4) + raw(value);
        ++count;
        return *this;
    }
    Gguf & boolean(const std::string & key, bool value) {
        metadata += str(key) + raw<uint32_t>(7) + raw<uint8_t>(value ? 1 : 0);
        ++count;
        return *this;
    }
    std::string bytes() const { return "GGUF" + raw<uint32_t>(3) + raw<uint64_t>(0) + raw(count) + metadata; }
};

Gguf model(const std::string & arch = "new_arch") {
    Gguf out;
    out.kv("general.architecture", arch).kv(arch + ".block_count", 32u).kv(arch + ".context_length", 32768u)
        .kv(arch + ".attention.head_count", 16u).kv(arch + ".attention.head_count_kv", 4u)
        .kv(arch + ".embedding_length", 4096u);
    return out;
}

std::map<std::string, std::string> headers;
const std::string listing =
    "[{\"path\":\"model-00001-of-00002.gguf\",\"size\":100000000},"
    "{\"path\":\"model-00002-of-00002.gguf\",\"size\":100000000},"
    "{\"path\":\"hidden-split.gguf\",\"size\":100000000},"
    "{\"path\":\"stripped.gguf\",\"size\":100000000},"
    "{\"path\":\"unrecognized.gguf\",\"size\":100000000},"
    "{\"path\":\"single.gguf\",\"size\":100000000}]";

void wait_for(const std::function<bool()> & done, const char * what) {
    for (int i = 0; i < 1000; ++i) {
        if (done()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::fprintf(stderr, "market_plan_test: timed out waiting for %s\n", what);
    std::abort();
}

int file_index(const ps5lm::MarketView & view, const std::string & name) {
    if (view.repos.empty()) {
        return -1;
    }
    const auto & files = view.repos[0].files;
    for (size_t i = 0; i < files.size(); ++i) {
        if (files[i].name == name) {
            return (int) i;
        }
    }
    return -1;
}

void wait_for_plan(ps5lm::Market & market, const std::string & name) {
    wait_for([&] {
        const auto view = market.snapshot();
        const int i = file_index(view, name);
        return i >= 0 && !view.repos[0].files[(size_t) i].checking;
    }, name.c_str());
}

}  // namespace

namespace ps5lm {

FetchResult https_get(const std::string & url, const std::string &, const std::function<bool(const char *, size_t)> & sink,
                      int, size_t) {
    FetchResult result;
    std::string body;
    if (url.find("/api/models?") != std::string::npos) {
        body = "[{\"id\":\"org/test\",\"downloads\":1}]";
    } else if (url.find("/api/models/org/test/tree/main") != std::string::npos) {
        body = listing;
    } else if (url.find("/resolve/main/") != std::string::npos) {
        const std::string name = url.substr(url.find_last_of('/') + 1);
        const auto it = headers.find(name);
        if (it == headers.end()) {
            result.status = 404;
            result.error = "not found in fake repository";
            return result;
        }
        body = it->second;
    } else {
        result.status = 404;
        result.error = "unrecognized fake URL";
        return result;
    }
    result.status = 200;
    result.length = (int64_t) body.size();
    result.bytes = body.size();
    if (!sink(body.data(), body.size())) {
        result.status = -1;
        result.error = "fake sink rejected body";
    }
    return result;
}

int https_template() { return -1; }

bool connect_ftp_writer(FtpWriter &, std::string & error) {
    error = "host test stub";
    return false;
}

bool FtpWriter::write_at(const std::string &, uint64_t, const char *, size_t, std::string & error) {
    error = "host test stub";
    return false;
}

void FtpWriter::close() {}

bool ftp_disk_full(const std::string &) { return false; }

}  // namespace ps5lm

int main() {
    headers["hidden-split.gguf"] = model().kv("split.count", 2u).kv("split.no", 0u).bytes();
    headers["stripped.gguf"] = Gguf{}.kv("general.architecture", "unknown").boolean("betelgeuse.stripped_experts", true).bytes();
    headers["unrecognized.gguf"] = Gguf{}.kv("general.architecture", "unrecognized_arch").bytes();
    headers["single.gguf"] = model().bytes();

    std::remove("build/market-test-queue.json");
    ps5lm::Market market;
    market.start(64.0);
    market.search("test");
    wait_for([&] { return !market.snapshot().busy; }, "search");
    market.open_repo(0);
    wait_for([&] { return !market.snapshot().busy; }, "repository listing");

    auto view = market.snapshot();
    const int first_shard = file_index(view, "model-00001-of-00002.gguf");
    assert(first_shard >= 0);
    const auto & split = view.repos[0].files[(size_t) first_shard];
    assert(split.unsupported && split.fits == -3);
    assert(split.plan.find("copy all shards locally") != std::string::npos);
    assert(file_index(view, "model-00002-of-00002.gguf") < 0);

    ps5lm::MarketJob job;
    assert(!market.job(0, first_shard, "build/models", &job));
    assert(!market.enqueue({ "org/test", "model-00001-of-00002.gguf", "", "build/models", 100000000, "" }, -1));

    const int hidden_split = file_index(view, "hidden-split.gguf");
    const int stripped = file_index(view, "stripped.gguf");
    const int unrecognized = file_index(view, "unrecognized.gguf");
    const int single = file_index(view, "single.gguf");
    assert(hidden_split >= 0 && stripped >= 0 && unrecognized >= 0 && single >= 0);
    assert(!market.job(0, unrecognized, "build/models", &job));  // do not race the header check

    market.plan_file(0, hidden_split);
    wait_for_plan(market, "hidden-split.gguf");
    view = market.snapshot();
    assert(view.repos[0].files[(size_t) hidden_split].unsupported);
    assert(view.repos[0].files[(size_t) hidden_split].fits == -3);
    assert(view.repos[0].files[(size_t) hidden_split].plan.find("all shards") != std::string::npos);
    assert(!market.job(0, hidden_split, "build/models", &job));
    assert(!market.enqueue({ "org/test", "hidden-split.gguf", "", "build/models", 100000000, "" }, -1));

    market.plan_file(0, stripped);
    wait_for_plan(market, "stripped.gguf");
    view = market.snapshot();
    assert(view.repos[0].files[(size_t) stripped].unsupported);
    assert(view.repos[0].files[(size_t) stripped].fits == -3);
    assert(view.repos[0].files[(size_t) stripped].plan.find("stripped experts") != std::string::npos);
    assert(!market.job(0, stripped, "build/models", &job));
    assert(!market.enqueue({ "org/test", "stripped.gguf", "", "build/models", 100000000, "" }, -1));

    market.plan_file(0, unrecognized);
    wait_for_plan(market, "unrecognized.gguf");
    view = market.snapshot();
    assert(!view.repos[0].files[(size_t) unrecognized].unsupported);
    assert(view.repos[0].files[(size_t) unrecognized].fits == -2);
    assert(market.job(0, unrecognized, "build/models", &job));
    assert(job.file == "unrecognized.gguf");

    market.plan_file(0, single);
    wait_for_plan(market, "single.gguf");
    view = market.snapshot();
    assert(!view.repos[0].files[(size_t) single].unsupported);
    assert(view.repos[0].files[(size_t) single].fits == 1);
    assert(market.job(0, single, "build/models", &job));

    // A split entry saved by an older version is failed at load and cannot be resumed.
    {
        std::ofstream queue("build/market-test-queue.json");
        queue << "[{\"repo\":\"org/test\",\"file\":\"old-00001-of-00002.gguf\",\"dir\":\"build/models\","
                 "\"bytes\":100,\"state\":\"queued\",\"resume\":0}]\n";
    }
    static ps5lm::Market recovered;
    recovered.load_queue();
    wait_for([&] {
        const auto saved = recovered.snapshot().downloads;
        return saved.size() == 1 && saved[0].state == ps5lm::DownloadState::failed;
    }, "saved split refusal");
    recovered.resume(0);
    const auto refused = recovered.snapshot().downloads[0];
    assert(refused.state == ps5lm::DownloadState::failed);
    assert(refused.note.find("copy all shards locally") != std::string::npos);

    // A curated suggested single-file job still enters the existing queue.
    static ps5lm::Market suggested;
    ps5lm::MarketJob curated{ "org/suggested", "curated.gguf", "", "build/market-suggested", 1024, "" };
    assert(suggested.enqueue(curated, -1));
    wait_for([&] {
        const auto downloads = suggested.snapshot().downloads;
        return downloads.size() == 1 && downloads[0].state == ps5lm::DownloadState::failed;
    }, "mocked curated download");
    assert(suggested.snapshot().downloads[0].note.find("Unsupported") == std::string::npos);

    std::puts("market_plan_test: ok (fake HTTPS only)");
}
