// Host test:
//   c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic ps5/moe/expert_source.cpp ps5/moe/slot_cache.cpp ps5/moe/paging_runtime.cpp ps5/moe/paging_runtime_test.cpp -o /tmp/ps5lm_paging_runtime_test
//   /tmp/ps5lm_paging_runtime_test

#include "paging_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ps5lm::moe;

namespace {

constexpr std::size_t kExpertBytes = 5;
constexpr std::size_t kStageBytes = 8;
constexpr std::uint32_t kLayer = 5;
constexpr std::uint32_t kOtherLayer = 6;
constexpr std::size_t kExpertCount = 3;

void expect(PagingStatus actual, PagingStatus wanted) {
    if (actual != wanted) {
        std::cerr << "expected " << paging_status_name(wanted) << ", got "
                  << paging_status_name(actual) << '\n';
        std::abort();
    }
}

void expect_cache(SlotCacheStatus actual, SlotCacheStatus wanted) {
    if (actual != wanted) {
        std::cerr << "expected cache " << slot_cache_status_name(wanted) << ", got "
                  << slot_cache_status_name(actual) << '\n';
        std::abort();
    }
}

struct TempDir {
    std::string path;
    std::string data_file;

    TempDir() {
        char pattern[] = "/tmp/ps5lm_paging_runtime_XXXXXX";
        char * created = ::mkdtemp(pattern);
        assert(created != nullptr);
        path = created;
        data_file = path + "/experts.gguf";
    }

    ~TempDir() {
        if (!data_file.empty()) {
            ::unlink(data_file.c_str());
        }
        if (!path.empty()) {
            ::rmdir(path.c_str());
        }
    }
};

std::uint64_t tensor_id(std::uint32_t layer, ExpertTensorRole role) {
    const std::uint64_t role_id = role == ExpertTensorRole::Gate ? 1 :
            role == ExpertTensorRole::Up ? 2 : 3;
    return 1000 + static_cast<std::uint64_t>(layer) * 10 + role_id;
}

std::uint8_t fixture_byte(std::uint32_t layer, ExpertTensorRole role, std::uint32_t expert, std::size_t byte) {
    const std::uint32_t role_id = role == ExpertTensorRole::Gate ? 11 :
            role == ExpertTensorRole::Up ? 37 : 73;
    return static_cast<std::uint8_t>((layer * 17 + role_id + expert * 29 + byte * 3) & 0xff);
}

struct Fixture {
    TempDir dir;
    std::vector<std::uint8_t> bytes;
    std::vector<ExpertTensorDescriptor> descriptors;
    OpenResult opened;

    Fixture() {
        std::uint64_t offset = 0;
        for (const std::uint32_t layer : {kLayer, kOtherLayer}) {
            for (const ExpertTensorRole role : {ExpertTensorRole::Gate, ExpertTensorRole::Up, ExpertTensorRole::Down}) {
                for (std::uint32_t expert = 0; expert < kExpertCount; ++expert) {
                    for (std::size_t byte = 0; byte < kExpertBytes; ++byte) {
                        bytes.push_back(fixture_byte(layer, role, expert, byte));
                    }
                }
                descriptors.push_back({
                    tensor_id(layer, role),
                    0,
                    offset,
                    kExpertCount * kExpertBytes,
                    kExpertCount,
                    kExpertBytes,
                    0,
                    {},
                });
                offset += kExpertCount * kExpertBytes;
            }
        }

        const int fd = ::open(dir.data_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        assert(fd >= 0);
        std::size_t written = 0;
        while (written < bytes.size()) {
            const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
            assert(count > 0);
            written += static_cast<std::size_t>(count);
        }
        assert(::close(fd) == 0);

        struct stat st{};
        assert(::stat(dir.data_file.c_str(), &st) == 0);
        const FileIdentity identity{st.st_ino != 0, static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
        for (ExpertTensorDescriptor & descriptor : descriptors) {
            descriptor.expected_file_size = bytes.size();
            descriptor.expected_identity = identity;
        }
        OpenOptions options;
        options.max_expert_bytes = kStageBytes;
        opened = ExpertSource::open({{dir.data_file}}, descriptors, options);
        assert(opened.ok());
    }

    std::vector<LayerExpertTensorIds> mappings() const {
        std::vector<LayerExpertTensorIds> result;
        for (const std::uint32_t layer : {kLayer, kOtherLayer}) {
            result.push_back({
                layer,
                {tensor_id(layer, ExpertTensorRole::Gate), kExpertBytes},
                {tensor_id(layer, ExpertTensorRole::Up), kExpertBytes},
                {tensor_id(layer, ExpertTensorRole::Down), kExpertBytes},
            });
        }
        return result;
    }

    std::vector<std::uint8_t> expected(std::uint32_t layer, ExpertTensorRole role, std::uint32_t expert) const {
        std::vector<std::uint8_t> result(kExpertBytes);
        for (std::size_t byte = 0; byte < kExpertBytes; ++byte) {
            result[byte] = fixture_byte(layer, role, expert, byte);
        }
        return result;
    }
};

SlotCacheOptions cache_options(std::size_t slots) {
    SlotCacheOptions result;
    result.slot_count = slots;
    result.bytes_per_slot = 3 * kExpertBytes;
    result.byte_budget = slots * result.bytes_per_slot;
    result.max_in_flight_tokens = 8;
    result.max_request_keys = 16;
    return result;
}

struct UploadKey {
    std::size_t slot;
    ExpertTensorRole role;
    friend bool operator<(const UploadKey & a, const UploadKey & b) noexcept {
        return std::tie(a.slot, a.role) < std::tie(b.slot, b.role);
    }
};

class FakeBackend final : public ExpertUploadBackend {
public:
    UploadSubmission submit_upload(
            ExpertKey key,
            ExpertTensorRole role,
            std::size_t slot,
            const void * source_bytes,
            std::size_t byte_count) noexcept override {
        ++submit_count;
        if (reject_on_submit != 0 && submit_count == reject_on_submit) {
            return {};
        }
        const UploadToken token = next_token++;
        pending.emplace(token, Pending{key, role, slot, static_cast<const std::uint8_t *>(source_bytes), byte_count});
        return {true, token};
    }

    TransferFenceStatus wait_transfer(UploadToken token) noexcept override {
        const auto found = pending.find(token);
        assert(found != pending.end());
        ++wait_count;
        if (wait_result.has_value()) {
            const TransferFenceStatus result = *wait_result;
            wait_result.reset();
            if (result == TransferFenceStatus::Indeterminate) {
                return result; // retain the raw staging pointer until quiesce
            }
            if (result == TransferFenceStatus::FailedButQuiesced) {
                pending.erase(found);
                return result;
            }
        }
        const Pending call = found->second;
        resident[{call.slot, call.role}] = std::vector<std::uint8_t>(call.source, call.source + call.bytes);
        pending.erase(found);
        if (cancel_on_successful_wait != nullptr) {
            cancel_on_successful_wait->store(true, std::memory_order_relaxed);
            cancel_on_successful_wait = nullptr;
        }
        return TransferFenceStatus::Complete;
    }

    TransferFenceStatus quiesce() noexcept override {
        ++quiesce_count;
        if (quiesce_result.has_value()) {
            const TransferFenceStatus result = *quiesce_result;
            if (result == TransferFenceStatus::Indeterminate) {
                return result;
            }
            pending.clear();
            quiesce_result.reset();
            return result;
        }
        pending.clear();
        return TransferFenceStatus::Complete;
    }

    const std::vector<std::uint8_t> * uploaded(std::size_t slot, ExpertTensorRole role) const {
        const auto found = resident.find({slot, role});
        return found == resident.end() ? nullptr : &found->second;
    }

    struct Pending {
        ExpertKey key;
        ExpertTensorRole role;
        std::size_t slot;
        const std::uint8_t * source;
        std::size_t bytes;
    };

    UploadToken next_token = 1;
    std::size_t submit_count = 0;
    std::size_t wait_count = 0;
    std::size_t quiesce_count = 0;
    std::size_t reject_on_submit = 0;
    std::optional<TransferFenceStatus> wait_result;
    std::optional<TransferFenceStatus> quiesce_result;
    std::atomic_bool * cancel_on_successful_wait = nullptr;
    std::map<UploadToken, Pending> pending;
    std::map<UploadKey, std::vector<std::uint8_t>> resident;
};

void test_exact_bytes_and_duplicate_ids() {
    Fixture fixture;
    SlotCache cache(cache_options(2));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);
    PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings(), 8);
    assert(runtime.configuration_status() == PagingStatus::Ok);

    PagingResult result = runtime.load_layer(kLayer, {0, 1, 0}, 1);
    expect(result.status, PagingStatus::Ok);
    assert(result.assignments.size() == 2);
    assert(backend.submit_count == 6 && backend.wait_count == 6);
    assert(backend.pending.empty());
    for (const SlotAssignment & assignment : result.assignments) {
        for (const ExpertTensorRole role : {ExpertTensorRole::Gate, ExpertTensorRole::Up, ExpertTensorRole::Down}) {
            const auto * actual = backend.uploaded(assignment.slot, role);
            assert(actual != nullptr);
            assert(*actual == fixture.expected(kLayer, role, assignment.key.expert));
        }
    }
    assert(cache.stats().resident_slots == 2 && cache.stats().pinned_slots == 2);
    expect_cache(cache.complete(1), SlotCacheStatus::Ok);

    result = runtime.load_layer(kOtherLayer, {2}, 2);
    expect(result.status, PagingStatus::Ok);
    assert(result.assignments.size() == 1);
    const std::size_t slot = result.assignments[0].slot;
    for (const ExpertTensorRole role : {ExpertTensorRole::Gate, ExpertTensorRole::Up, ExpertTensorRole::Down}) {
        const auto * actual = backend.uploaded(slot, role);
        assert(actual != nullptr);
        assert(*actual == fixture.expected(kOtherLayer, role, 2));
    }
    expect_cache(cache.complete(2), SlotCacheStatus::Ok);
}

void test_rejects_bad_configuration_and_bounds_requests() {
    Fixture fixture;
    SlotCache cache(cache_options(2));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);
    auto map = fixture.mappings();
    map[0].up.bytes_per_expert = kStageBytes + 1;
    PagingRuntime invalid(cache, *fixture.opened.source, backend, quarantine, map);
    assert(invalid.configuration_status() == PagingStatus::InvalidConfiguration);

    PagingQuarantine small_quarantine(kExpertBytes - 1);
    PagingRuntime too_small(cache, *fixture.opened.source, backend, small_quarantine, fixture.mappings());
    assert(too_small.configuration_status() == PagingStatus::InvalidConfiguration);

    // The failed constructors never acquire the buffer lease.
    PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings(), 2);
    expect(runtime.load_layer(999, {0}, 1).status, PagingStatus::InvalidLayer);
    expect(runtime.load_layer(kLayer, {0, 1, 2}, 1).status, PagingStatus::RequestTooLarge);
    expect(runtime.load_layer(kLayer, {}, 1).status, PagingStatus::CacheFailure);
}

void test_cancel_and_safe_upload_failures_roll_back() {
    Fixture fixture;
    SlotCache cache(cache_options(1));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);
    PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings());

    std::atomic_bool cancelled{true};
    expect(runtime.load_layer(kLayer, {0}, 1, &cancelled).status, PagingStatus::Cancelled);
    assert(backend.submit_count == 0 && cache.stats().reserved_slots == 0);

    cancelled.store(false, std::memory_order_relaxed);
    backend.cancel_on_successful_wait = &cancelled;
    PagingResult result = runtime.load_layer(kLayer, {0}, 1, &cancelled);
    expect(result.status, PagingStatus::Cancelled);
    assert(backend.submit_count == 1 && backend.wait_count == 1);
    assert(backend.pending.empty());
    assert(cache.stats().resident_slots == 0 && cache.stats().reserved_slots == 0);
    std::size_t slot = kNoSlot;
    assert(!cache.find_slot({kLayer, 0}, &slot));

    backend.reject_on_submit = backend.submit_count + 1;
    result = runtime.load_layer(kLayer, {1}, 1);
    expect(result.status, PagingStatus::UploadRejected);
    assert(cache.stats().resident_slots == 0 && cache.stats().reserved_slots == 0);

    backend.reject_on_submit = 0;
    backend.wait_result = TransferFenceStatus::FailedButQuiesced;
    result = runtime.load_layer(kLayer, {1}, 1);
    expect(result.status, PagingStatus::UploadFailed);
    assert(backend.pending.empty());
    assert(cache.stats().resident_slots == 0 && cache.stats().reserved_slots == 0);

    result = runtime.load_layer(kLayer, {1}, 1);
    expect(result.status, PagingStatus::Ok);
    expect_cache(cache.complete(1), SlotCacheStatus::Ok);
}

void test_source_failure_does_not_publish_or_upload() {
    Fixture fixture;
    SlotCache cache(cache_options(1));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);
    PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings());

    assert(::truncate(fixture.dir.data_file.c_str(), static_cast<off_t>(fixture.bytes.size() - 1)) == 0);
    PagingResult result = runtime.load_layer(kLayer, {0}, 1);
    expect(result.status, PagingStatus::SourceFailure);
    assert(result.read_error.status == ReadStatus::FileChanged);
    assert(backend.submit_count == 0);
    assert(cache.stats().reserved_slots == 0 && cache.stats().resident_slots == 0);
    std::size_t slot = kNoSlot;
    assert(!cache.find_slot({kLayer, 0}, &slot));
}

void test_pins_eviction_and_token_errors() {
    Fixture fixture;
    SlotCache cache(cache_options(2));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);
    PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings());

    expect(runtime.load_layer(kLayer, {0}, 1).status, PagingStatus::Ok);
    expect(runtime.load_layer(kLayer, {1}, 2).status, PagingStatus::Ok);
    PagingResult blocked = runtime.load_layer(kLayer, {2}, 3);
    expect(blocked.status, PagingStatus::CacheFailure);
    expect_cache(blocked.cache_status, SlotCacheStatus::InsufficientUnpinnedSlots);
    assert(backend.submit_count == 6);

    expect_cache(cache.complete(1), SlotCacheStatus::Ok);
    PagingResult evicted = runtime.load_layer(kLayer, {2}, 3);
    expect(evicted.status, PagingStatus::Ok);
    std::size_t slot = kNoSlot;
    assert(!cache.find_slot({kLayer, 0}, &slot));
    assert(cache.find_slot({kLayer, 1}, &slot));
    assert(cache.find_slot({kLayer, 2}, &slot));
    expect_cache(cache.complete(2), SlotCacheStatus::Ok);
    expect_cache(cache.complete(3), SlotCacheStatus::Ok);

    PagingResult bad_token = runtime.load_layer(kLayer, {1}, 3);
    expect(bad_token.status, PagingStatus::CacheFailure);
    expect_cache(bad_token.cache_status, SlotCacheStatus::InvalidCompletionToken);
    assert(!cache.stats().active_reservations);
}

void test_indeterminate_fence_quarantines_slot_and_staging_until_quiesced() {
    Fixture fixture;
    SlotCache cache(cache_options(1));
    FakeBackend backend;
    PagingQuarantine quarantine(kStageBytes);

    {
        PagingRuntime runtime(cache, *fixture.opened.source, backend, quarantine, fixture.mappings());
        backend.wait_result = TransferFenceStatus::Indeterminate;
        PagingResult result = runtime.load_layer(kLayer, {0}, 1);
        expect(result.status, PagingStatus::FenceUnresolved);
        assert(runtime.poisoned() && quarantine.pending());
        assert(quarantine.pending_transfer() == result.transfer_token);
        const ExpertKey expected_key{kLayer, 0};
        assert(quarantine.pending_key() == expected_key);
        assert(quarantine.pending_role() == ExpertTensorRole::Gate);
        assert(backend.pending.size() == 1);
        std::size_t slot = kNoSlot;
        assert(!cache.find_slot({kLayer, 0}, &slot));
        auto cannot_reuse = cache.reserve({{kOtherLayer, 1}});
        expect_cache(cannot_reuse.status, SlotCacheStatus::InsufficientUnpinnedSlots);
        expect(runtime.load_layer(kOtherLayer, {1}, 2).status, PagingStatus::Quarantined);
    }

    backend.quiesce_result = TransferFenceStatus::Indeterminate;
    assert(quarantine.release_after_backend_quiesced(backend) == QuarantineStatus::StillUnsafe);
    assert(quarantine.pending() && !backend.pending.empty());
    FakeBackend wrong_backend;
    assert(quarantine.release_after_backend_quiesced(wrong_backend) == QuarantineStatus::StillUnsafe);
    assert(wrong_backend.quiesce_count == 0);
    auto still_blocked = cache.reserve({{kOtherLayer, 1}});
    expect_cache(still_blocked.status, SlotCacheStatus::InsufficientUnpinnedSlots);

    backend.quiesce_result = TransferFenceStatus::Complete;
    assert(quarantine.release_after_backend_quiesced(backend) == QuarantineStatus::Released);
    assert(!quarantine.pending() && backend.pending.empty());
    auto available = cache.reserve({{kOtherLayer, 1}});
    expect_cache(available.status, SlotCacheStatus::Ok);
    available.reservation.rollback();
}

}  // namespace

int main() {
    test_exact_bytes_and_duplicate_ids();
    test_rejects_bad_configuration_and_bounds_requests();
    test_cancel_and_safe_upload_failures_roll_back();
    test_source_failure_does_not_publish_or_upload();
    test_pins_eviction_and_token_errors();
    test_indeterminate_fence_quarantines_slot_and_staging_until_quiesced();
    std::cout << "paging_runtime_test: PASS\n";
}
