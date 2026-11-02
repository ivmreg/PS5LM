// Synchronous per-layer adapter between ExpertSource and SlotCache.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "expert_source.hpp"
#include "slot_cache.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace ps5lm::moe {

using UploadToken = std::uint64_t;

enum class ExpertTensorRole : std::uint8_t { Gate, Up, Down };

struct ExpertTensorRef {
    std::uint64_t tensor_id = 0;
    // Exact per-expert stride from the same validated metadata used to build
    // ExpertSource. ExpertSource::read_expert does not return a byte count.
    std::size_t bytes_per_expert = 0;
};

struct LayerExpertTensorIds {
    std::uint32_t layer = 0;
    ExpertTensorRef gate{};
    ExpertTensorRef up{};
    ExpertTensorRef down{};
};

enum class TransferFenceStatus : std::uint8_t {
    Complete,
    FailedButQuiesced,
    Indeterminate,
};

struct UploadSubmission {
    // false guarantees that no work was accepted and the source pointer is no
    // longer retained. true requires a valid token unless the caller will
    // quarantine and globally quiesce the backend.
    bool submitted = false;
    UploadToken token = 0;
};

// Backend contract: submit_upload() may retain source_bytes only while its
// returned token is outstanding. wait_transfer() must return Complete or
// FailedButQuiesced only when it guarantees no further source/destination
// access. Indeterminate means the caller cannot reuse either resource.
class ExpertUploadBackend {
public:
    virtual ~ExpertUploadBackend() = default;

    virtual UploadSubmission submit_upload(
            ExpertKey key,
            ExpertTensorRole role,
            std::size_t slot,
            const void * source_bytes,
            std::size_t byte_count) noexcept = 0;

    virtual TransferFenceStatus wait_transfer(UploadToken token) noexcept = 0;

    // Stops/drains all accepted transfers. Only Complete or FailedButQuiesced
    // confirms that no outstanding transfer can access staging or cache slots.
    virtual TransferFenceStatus quiesce() noexcept = 0;
};

enum class PagingStatus : std::uint8_t {
    Ok,
    InvalidConfiguration,
    InvalidLayer,
    RequestTooLarge,
    Cancelled,
    CacheFailure,
    SourceFailure,
    UploadRejected,
    UploadFailed,
    FenceUnresolved,
    Quarantined,
    OutOfMemory,
};

const char * paging_status_name(PagingStatus status) noexcept;

struct PagingResult {
    PagingStatus status = PagingStatus::InvalidConfiguration;
    SlotCacheStatus cache_status = SlotCacheStatus::Ok;
    ReadError read_error{};
    ExpertKey failed_key{};
    ExpertTensorRole failed_role = ExpertTensorRole::Gate;
    UploadToken transfer_token = 0;
    CompletionToken gpu_use_token = 0;
    std::vector<SlotAssignment> assignments;

    bool ok() const noexcept { return status == PagingStatus::Ok; }
};

enum class QuarantineStatus : std::uint8_t {
    Released,
    NotPending,
    StillUnsafe,
};

// Owns exactly one fixed reusable staging buffer. If a transfer fence becomes
// indeterminate it also owns the active reservation, preventing slot rollback
// and staging destruction until backend-wide quiescence is confirmed.
class PagingQuarantine final {
public:
    explicit PagingQuarantine(std::size_t staging_capacity);
    ~PagingQuarantine() noexcept;
    PagingQuarantine(const PagingQuarantine &) = delete;
    PagingQuarantine & operator=(const PagingQuarantine &) = delete;

    std::size_t staging_capacity() const noexcept { return staging_.size(); }
    bool pending() const noexcept { return pending_.has_value(); }
    UploadToken pending_transfer() const noexcept { return pending_transfer_; }
    ExpertKey pending_key() const noexcept { return pending_key_; }
    ExpertTensorRole pending_role() const noexcept { return pending_role_; }
    std::size_t pending_slot() const noexcept { return pending_slot_; }

    // This method calls backend.quiesce(). It releases the reservation only
    // after a result that guarantees no transfer remains in flight.
    QuarantineStatus release_after_backend_quiesced(ExpertUploadBackend & backend) noexcept;

private:
    friend class PagingRuntime;

    bool acquire(ExpertUploadBackend & backend) noexcept;
    void release() noexcept;
    void quarantine(
            Reservation && reservation,
            ExpertKey key,
            ExpertTensorRole role,
            std::size_t slot,
            UploadToken transfer) noexcept;

    std::vector<std::uint8_t> staging_;
    std::optional<Reservation> pending_;
    bool leased_ = false;
    ExpertUploadBackend * backend_ = nullptr;
    UploadToken pending_transfer_ = 0;
    ExpertKey pending_key_{};
    ExpertTensorRole pending_role_ = ExpertTensorRole::Gate;
    std::size_t pending_slot_ = kNoSlot;
};

class PagingRuntime final {
public:
    static constexpr std::size_t kHardLayerLimit = 1024;
    static constexpr std::size_t kHardSelectedExpertLimit = 256;

    PagingRuntime(
            SlotCache & cache,
            const ExpertSource & source,
            ExpertUploadBackend & backend,
            PagingQuarantine & quarantine,
            const std::vector<LayerExpertTensorIds> & tensor_ids,
            std::size_t max_selected_experts = 64);
    ~PagingRuntime() noexcept;
    PagingRuntime(const PagingRuntime &) = delete;
    PagingRuntime & operator=(const PagingRuntime &) = delete;

    PagingStatus configuration_status() const noexcept { return configuration_status_; }
    bool poisoned() const noexcept { return poisoned_; }

    // Uploads every selected expert's gate, up and down slices for one layer.
    // Duplicate expert IDs consume one cache slot and one set of role reads.
    // The caller must not submit GPU compute using assignments until this
    // returns Ok; gpu_use_token pins the slots until SlotCache::complete().
    PagingResult load_layer(
            std::uint32_t layer,
            const std::vector<std::uint32_t> & selected_experts,
            CompletionToken gpu_use_token,
            const std::atomic_bool * cancelled = nullptr) noexcept;

private:
    const LayerExpertTensorIds * find_layer(std::uint32_t layer) const noexcept;
    bool is_cancelled(const std::atomic_bool * cancelled) const noexcept;
    PagingResult fail_with_cache(PagingResult result, SlotCacheStatus status) const noexcept;

    SlotCache & cache_;
    const ExpertSource & source_;
    ExpertUploadBackend & backend_;
    PagingQuarantine & quarantine_;
    std::map<std::uint32_t, LayerExpertTensorIds> tensor_ids_;
    std::size_t max_selected_experts_ = 0;
    PagingStatus configuration_status_ = PagingStatus::InvalidConfiguration;
    bool leased_ = false;
    bool poisoned_ = false;
};

}  // namespace ps5lm::moe
