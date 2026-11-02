// Backend-independent transactional slot policy for MoE expert paging.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace ps5lm::moe {

struct ExpertKey {
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;

    friend bool operator==(ExpertKey a, ExpertKey b) noexcept {
        return a.layer == b.layer && a.expert == b.expert;
    }
    friend bool operator!=(ExpertKey a, ExpertKey b) noexcept { return !(a == b); }
};

using CompletionToken = std::uint64_t;
constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

enum class SlotCacheStatus : std::uint8_t {
    Ok,
    InvalidConfiguration,
    BudgetExceeded,
    TooManySlots,
    OutOfMemory,
    InvalidRequest,
    RequestTooLarge,
    TooManyUniqueExperts,
    TooManyOutstandingBatches,
    KeyReserved,
    InsufficientUnpinnedSlots,
    InvalidReservation,
    UploadNotRequired,
    UploadsIncomplete,
    InvalidCompletionToken,
    TooManyInFlightTokens,
    UnknownCompletionToken,
    InternalError,
};

const char * slot_cache_status_name(SlotCacheStatus status) noexcept;

struct SlotCacheOptions {
    std::size_t slot_count = 0;
    std::size_t bytes_per_slot = 0;
    std::size_t byte_budget = 0;
    std::size_t max_in_flight_tokens = 16;
    std::size_t max_request_keys = 4096;
};

// One entry for each distinct requested identity. Repeated request IDs resolve
// through slot_for() to this same assignment and never consume extra slots.
struct SlotAssignment {
    ExpertKey key{};
    std::size_t slot = kNoSlot;
    bool needs_upload = false;
    bool upload_complete = false;
};

struct SlotCacheStats {
    std::size_t capacity_slots = 0;
    std::size_t resident_slots = 0;
    std::size_t reserved_slots = 0;
    std::size_t pinned_slots = 0;
    std::size_t active_reservations = 0;
    std::size_t active_completion_tokens = 0;
    std::size_t bytes_per_slot = 0;
    std::size_t configured_capacity_bytes = 0;
    std::size_t byte_budget = 0;
    std::size_t metadata_bytes = 0;
};

struct SlotView {
    bool resident = false;
    bool reserved = false;
    ExpertKey key{};  // resident key or pending reservation key
    std::size_t outstanding_reservations = 0;
    std::size_t outstanding_gpu_uses = 0;
};

class SlotCache;

class Reservation final {
public:
    Reservation() noexcept = default;
    ~Reservation() noexcept;
    Reservation(const Reservation &) = delete;
    Reservation & operator=(const Reservation &) = delete;
    Reservation(Reservation && other) noexcept;
    Reservation & operator=(Reservation && other) noexcept;

    bool active() const noexcept { return active_; }
    const std::vector<SlotAssignment> & assignments() const noexcept { return assignments_; }
    bool slot_for(ExpertKey key, std::size_t * slot_out) const noexcept;

    // Call only after that key's read and upload have both completed. It is
    // idempotent for a key that was already marked complete.
    SlotCacheStatus mark_upload_complete(ExpertKey key) noexcept;

    // Atomically publishes every newly uploaded key and pins all batch slots
    // to this unique, monotonically increasing future GPU completion token.
    // Uploads incomplete leaves the reservation active so the caller can
    // finish or roll it back.
    SlotCacheStatus commit(CompletionToken token) noexcept;

    // Cancellation, read failure, or upload failure invalidates all pending
    // slots and releases hit reservations. The handle's destructor rolls back
    // an active reservation automatically.
    void rollback() noexcept;

private:
    friend class SlotCache;
    Reservation(SlotCache * owner, std::uint64_t id, std::vector<SlotAssignment> assignments) noexcept;

    SlotCache * owner_ = nullptr;  // SlotCache must outlive its reservations
    std::uint64_t id_ = 0;
    bool active_ = false;
    std::vector<SlotAssignment> assignments_;
};

struct ReserveResult {
    SlotCacheStatus status = SlotCacheStatus::InvalidRequest;
    Reservation reservation;
    bool ok() const noexcept { return status == SlotCacheStatus::Ok && reservation.active(); }
};

class SlotCache final {
public:
    static constexpr std::size_t kHardSlotLimit = 4096;
    static constexpr std::size_t kHardRequestKeyLimit = 4096;
    static constexpr std::size_t kHardInFlightLimit = 256;

    explicit SlotCache(const SlotCacheOptions & options) noexcept;
    ~SlotCache() noexcept = default;
    SlotCache(const SlotCache &) = delete;
    SlotCache & operator=(const SlotCache &) = delete;

    SlotCacheStatus configuration_status() const noexcept { return configuration_status_; }
    std::size_t slot_count() const noexcept { return slots_.size(); }

    // Reservation is all-or-nothing. It deduplicates repeated identities,
    // protects hits and reserves enough empty/LRU-unpinned slots for every miss.
    ReserveResult reserve(const std::vector<ExpertKey> & requested);

    // Completes a fake/real timeline token. Only this operation releases the
    // corresponding GPU-use pins. Unknown and repeated completion signals fail.
    SlotCacheStatus complete(CompletionToken token) noexcept;

    bool find_slot(ExpertKey key, std::size_t * slot_out) const noexcept;
    bool inspect_slot(std::size_t index, SlotView * view_out) const noexcept;
    SlotCacheStats stats() const noexcept;

private:
    friend class Reservation;

    enum class State : std::uint8_t { Empty, Resident, Reserved };
    struct Slot {
        State state = State::Empty;
        ExpertKey key{};
        std::size_t reservation_refs = 0;
        std::size_t gpu_use_refs = 0;
        std::uint64_t pending_transaction = 0;
        std::uint64_t last_use = 0;
    };

    SlotCacheStatus commit(Reservation & reservation, CompletionToken token) noexcept;
    void rollback(Reservation & reservation) noexcept;
    std::size_t find_resident_locked(ExpertKey key) const noexcept;
    std::size_t find_reserved_locked(ExpertKey key) const noexcept;
    std::uint64_t next_lru_stamp() noexcept;

    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::map<CompletionToken, std::vector<std::size_t>> token_slots_;
    SlotCacheOptions options_{};
    SlotCacheStatus configuration_status_ = SlotCacheStatus::InvalidConfiguration;
    std::size_t configured_capacity_bytes_ = 0;
    std::size_t active_reservations_ = 0;
    std::uint64_t next_transaction_id_ = 1;
    std::uint64_t last_completion_token_ = 0;
    std::uint64_t lru_clock_ = 0;
};

}  // namespace ps5lm::moe
