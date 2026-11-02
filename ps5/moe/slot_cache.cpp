// Backend-independent transactional slot policy for MoE expert paging.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "slot_cache.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace ps5lm::moe {
namespace {

bool checked_multiply(std::size_t a, std::size_t b, std::size_t * out) noexcept {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

}  // namespace

const char * slot_cache_status_name(SlotCacheStatus status) noexcept {
    switch (status) {
        case SlotCacheStatus::Ok: return "ok";
        case SlotCacheStatus::InvalidConfiguration: return "invalid_configuration";
        case SlotCacheStatus::BudgetExceeded: return "budget_exceeded";
        case SlotCacheStatus::TooManySlots: return "too_many_slots";
        case SlotCacheStatus::OutOfMemory: return "out_of_memory";
        case SlotCacheStatus::InvalidRequest: return "invalid_request";
        case SlotCacheStatus::RequestTooLarge: return "request_too_large";
        case SlotCacheStatus::TooManyUniqueExperts: return "too_many_unique_experts";
        case SlotCacheStatus::TooManyOutstandingBatches: return "too_many_outstanding_batches";
        case SlotCacheStatus::KeyReserved: return "key_reserved";
        case SlotCacheStatus::InsufficientUnpinnedSlots: return "insufficient_unpinned_slots";
        case SlotCacheStatus::InvalidReservation: return "invalid_reservation";
        case SlotCacheStatus::UploadNotRequired: return "upload_not_required";
        case SlotCacheStatus::UploadsIncomplete: return "uploads_incomplete";
        case SlotCacheStatus::InvalidCompletionToken: return "invalid_completion_token";
        case SlotCacheStatus::TooManyInFlightTokens: return "too_many_in_flight_tokens";
        case SlotCacheStatus::UnknownCompletionToken: return "unknown_completion_token";
        case SlotCacheStatus::InternalError: return "internal_error";
    }
    return "unknown";
}

Reservation::Reservation(
        SlotCache * owner,
        std::uint64_t id,
        std::vector<SlotAssignment> assignments) noexcept
    : owner_(owner), id_(id), active_(owner != nullptr), assignments_(std::move(assignments)) {}

Reservation::~Reservation() noexcept {
    rollback();
}

Reservation::Reservation(Reservation && other) noexcept
    : owner_(other.owner_),
      id_(other.id_),
      active_(other.active_),
      assignments_(std::move(other.assignments_)) {
    other.owner_ = nullptr;
    other.id_ = 0;
    other.active_ = false;
}

Reservation & Reservation::operator=(Reservation && other) noexcept {
    if (this != &other) {
        rollback();
        owner_ = other.owner_;
        id_ = other.id_;
        active_ = other.active_;
        assignments_ = std::move(other.assignments_);
        other.owner_ = nullptr;
        other.id_ = 0;
        other.active_ = false;
    }
    return *this;
}

bool Reservation::slot_for(ExpertKey key, std::size_t * slot_out) const noexcept {
    if (!active_ || slot_out == nullptr) {
        return false;
    }
    for (const SlotAssignment & assignment : assignments_) {
        if (assignment.key == key) {
            *slot_out = assignment.slot;
            return true;
        }
    }
    return false;
}

SlotCacheStatus Reservation::mark_upload_complete(ExpertKey key) noexcept {
    if (!active_ || owner_ == nullptr) {
        return SlotCacheStatus::InvalidReservation;
    }
    for (SlotAssignment & assignment : assignments_) {
        if (assignment.key == key) {
            if (!assignment.needs_upload) {
                return SlotCacheStatus::UploadNotRequired;
            }
            assignment.upload_complete = true;
            return SlotCacheStatus::Ok;
        }
    }
    return SlotCacheStatus::InvalidRequest;
}

SlotCacheStatus Reservation::commit(CompletionToken token) noexcept {
    if (!active_ || owner_ == nullptr) {
        return SlotCacheStatus::InvalidReservation;
    }
    return owner_->commit(*this, token);
}

void Reservation::rollback() noexcept {
    if (active_ && owner_ != nullptr) {
        owner_->rollback(*this);
    }
}

SlotCache::SlotCache(const SlotCacheOptions & options) noexcept : options_(options) {
    if (options.slot_count == 0 || options.bytes_per_slot == 0 || options.byte_budget == 0 ||
        options.max_request_keys == 0 || options.max_in_flight_tokens == 0) {
        configuration_status_ = SlotCacheStatus::InvalidConfiguration;
        return;
    }
    if (options.slot_count > kHardSlotLimit) {
        configuration_status_ = SlotCacheStatus::TooManySlots;
        return;
    }
    if (options.max_request_keys > kHardRequestKeyLimit ||
        options.max_in_flight_tokens > kHardInFlightLimit) {
        configuration_status_ = SlotCacheStatus::InvalidConfiguration;
        return;
    }
    if (!checked_multiply(options.slot_count, options.bytes_per_slot, &configured_capacity_bytes_)) {
        configuration_status_ = SlotCacheStatus::InvalidConfiguration;
        return;
    }
    if (configured_capacity_bytes_ > options.byte_budget) {
        configuration_status_ = SlotCacheStatus::BudgetExceeded;
        return;
    }
    try {
        slots_.resize(options.slot_count);
    } catch (const std::bad_alloc &) {
        configuration_status_ = SlotCacheStatus::OutOfMemory;
        return;
    } catch (...) {
        configuration_status_ = SlotCacheStatus::OutOfMemory;
        return;
    }
    configuration_status_ = SlotCacheStatus::Ok;
}

std::size_t SlotCache::find_resident_locked(ExpertKey key) const noexcept {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const Slot & slot = slots_[i];
        if (slot.state == State::Resident && slot.key == key) {
            return i;
        }
    }
    return kNoSlot;
}

std::size_t SlotCache::find_reserved_locked(ExpertKey key) const noexcept {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const Slot & slot = slots_[i];
        if (slot.state == State::Reserved && slot.key == key) {
            return i;
        }
    }
    return kNoSlot;
}

std::uint64_t SlotCache::next_lru_stamp() noexcept {
    if (lru_clock_ != std::numeric_limits<std::uint64_t>::max()) {
        return ++lru_clock_;
    }
    // This is only reachable after 2^64 cache touches. Saturating keeps the
    // operation defined; ties are then resolved by the lowest slot index.
    return lru_clock_;
}

ReserveResult SlotCache::reserve(const std::vector<ExpertKey> & requested) {
    ReserveResult result;
    result.status = configuration_status_;
    if (configuration_status_ != SlotCacheStatus::Ok) {
        return result;
    }
    if (requested.empty()) {
        result.status = SlotCacheStatus::InvalidRequest;
        return result;
    }
    if (requested.size() > options_.max_request_keys) {
        result.status = SlotCacheStatus::RequestTooLarge;
        return result;
    }

    std::vector<SlotAssignment> assignments;
    std::vector<bool> protected_slots;
    try {
        assignments.reserve(requested.size());
        protected_slots.resize(slots_.size(), false);
        for (const ExpertKey key : requested) {
            const auto found = std::find_if(assignments.begin(), assignments.end(),
                    [key](const SlotAssignment & entry) { return entry.key == key; });
            if (found == assignments.end()) {
                assignments.push_back({key, kNoSlot, false, false});
            }
        }
    } catch (...) {
        result.status = SlotCacheStatus::OutOfMemory;
        return result;
    }
    if (assignments.size() > slots_.size()) {
        result.status = SlotCacheStatus::TooManyUniqueExperts;
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (active_reservations_ >= kHardInFlightLimit) {
        result.status = SlotCacheStatus::TooManyOutstandingBatches;
        return result;
    }

    // Resolve all hits first. A matching in-progress upload conflicts with
    // the entire request; no partial slot state is changed in this pass.
    for (SlotAssignment & assignment : assignments) {
        const std::size_t resident = find_resident_locked(assignment.key);
        if (resident != kNoSlot) {
            assignment.slot = resident;
            protected_slots[resident] = true;
            continue;
        }
        if (find_reserved_locked(assignment.key) != kNoSlot) {
            result.status = SlotCacheStatus::KeyReserved;
            return result;
        }
        assignment.needs_upload = true;
    }

    // Plan every miss before mutating the cache. Empty slots are preferred;
    // otherwise choose the oldest resident slot that has no reservation or
    // GPU-use pins. Hits and earlier candidates are excluded.
    for (SlotAssignment & assignment : assignments) {
        if (!assignment.needs_upload) {
            continue;
        }
        std::size_t candidate = kNoSlot;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            if (!protected_slots[i] && slots_[i].state == State::Empty) {
                candidate = i;
                break;
            }
        }
        if (candidate == kNoSlot) {
            std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (protected_slots[i] || slot.state != State::Resident ||
                    slot.reservation_refs != 0 || slot.gpu_use_refs != 0) {
                    continue;
                }
                if (candidate == kNoSlot || slot.last_use < oldest) {
                    candidate = i;
                    oldest = slot.last_use;
                }
            }
        }
        if (candidate == kNoSlot) {
            result.status = SlotCacheStatus::InsufficientUnpinnedSlots;
            return result;
        }
        assignment.slot = candidate;
        protected_slots[candidate] = true;
    }

    std::uint64_t id = next_transaction_id_++;
    if (id == 0) {
        id = next_transaction_id_++;
    }
    if (next_transaction_id_ == 0) {
        next_transaction_id_ = 1;
    }
    for (const SlotAssignment & assignment : assignments) {
        Slot & slot = slots_[assignment.slot];
        if (assignment.needs_upload) {
            slot.state = State::Reserved;
            slot.key = assignment.key;
            slot.reservation_refs = 0;
            slot.gpu_use_refs = 0;
            slot.pending_transaction = id;
            slot.last_use = next_lru_stamp();
        } else {
            ++slot.reservation_refs;
        }
    }
    ++active_reservations_;
    result.status = SlotCacheStatus::Ok;
    result.reservation = Reservation(this, id, std::move(assignments));
    return result;
}

SlotCacheStatus SlotCache::commit(Reservation & reservation, CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reservation.active_ || reservation.owner_ != this || reservation.id_ == 0) {
        return SlotCacheStatus::InvalidReservation;
    }
    if (token == 0 || token <= last_completion_token_) {
        return SlotCacheStatus::InvalidCompletionToken;
    }
    if (token_slots_.size() >= options_.max_in_flight_tokens) {
        return SlotCacheStatus::TooManyInFlightTokens;
    }
    for (const SlotAssignment & assignment : reservation.assignments_) {
        if (assignment.needs_upload && !assignment.upload_complete) {
            return SlotCacheStatus::UploadsIncomplete;
        }
        if (assignment.slot >= slots_.size()) {
            return SlotCacheStatus::InternalError;
        }
        const Slot & slot = slots_[assignment.slot];
        if (assignment.needs_upload) {
            if (slot.state != State::Reserved || slot.pending_transaction != reservation.id_ ||
                slot.key != assignment.key) {
                return SlotCacheStatus::InvalidReservation;
            }
        } else if (slot.state != State::Resident || slot.key != assignment.key ||
                   slot.reservation_refs == 0) {
            return SlotCacheStatus::InvalidReservation;
        }
    }

    std::vector<std::size_t> pinned_slots;
    try {
        pinned_slots.reserve(reservation.assignments_.size());
        for (const SlotAssignment & assignment : reservation.assignments_) {
            pinned_slots.push_back(assignment.slot);
        }
        const auto inserted = token_slots_.emplace(token, std::move(pinned_slots));
        if (!inserted.second) {
            return SlotCacheStatus::InvalidCompletionToken;
        }
    } catch (...) {
        return SlotCacheStatus::OutOfMemory;
    }

    for (const SlotAssignment & assignment : reservation.assignments_) {
        Slot & slot = slots_[assignment.slot];
        if (assignment.needs_upload) {
            slot.state = State::Resident;
            slot.pending_transaction = 0;
        } else {
            --slot.reservation_refs;
        }
        ++slot.gpu_use_refs;
        slot.last_use = next_lru_stamp();
    }
    last_completion_token_ = token;
    if (active_reservations_ != 0) {
        --active_reservations_;
    }
    reservation.owner_ = nullptr;
    reservation.id_ = 0;
    reservation.active_ = false;
    reservation.assignments_.clear();
    return SlotCacheStatus::Ok;
}

void SlotCache::rollback(Reservation & reservation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reservation.active_ || reservation.owner_ != this || reservation.id_ == 0) {
        reservation.owner_ = nullptr;
        reservation.id_ = 0;
        reservation.active_ = false;
        reservation.assignments_.clear();
        return;
    }
    for (const SlotAssignment & assignment : reservation.assignments_) {
        if (assignment.slot >= slots_.size()) {
            continue;
        }
        Slot & slot = slots_[assignment.slot];
        if (assignment.needs_upload) {
            if (slot.state == State::Reserved && slot.pending_transaction == reservation.id_) {
                slot = Slot{};
            }
        } else if (slot.state == State::Resident && slot.key == assignment.key &&
                   slot.reservation_refs != 0) {
            --slot.reservation_refs;
        }
    }
    if (active_reservations_ != 0) {
        --active_reservations_;
    }
    reservation.owner_ = nullptr;
    reservation.id_ = 0;
    reservation.active_ = false;
    reservation.assignments_.clear();
}

SlotCacheStatus SlotCache::complete(CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = token_slots_.find(token);
    if (found == token_slots_.end()) {
        return SlotCacheStatus::UnknownCompletionToken;
    }
    for (const std::size_t index : found->second) {
        if (index >= slots_.size() || slots_[index].gpu_use_refs == 0) {
            return SlotCacheStatus::InternalError;
        }
    }
    for (const std::size_t index : found->second) {
        --slots_[index].gpu_use_refs;
    }
    token_slots_.erase(found);
    return SlotCacheStatus::Ok;
}

bool SlotCache::find_slot(ExpertKey key, std::size_t * slot_out) const noexcept {
    if (slot_out == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t found = find_resident_locked(key);
    if (found == kNoSlot) {
        return false;
    }
    *slot_out = found;
    return true;
}

bool SlotCache::inspect_slot(std::size_t index, SlotView * view_out) const noexcept {
    if (view_out == nullptr || index >= slots_.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const Slot & slot = slots_[index];
    view_out->resident = slot.state == State::Resident;
    view_out->reserved = slot.state == State::Reserved;
    view_out->key = slot.key;
    view_out->outstanding_reservations = slot.reservation_refs;
    view_out->outstanding_gpu_uses = slot.gpu_use_refs;
    return true;
}

SlotCacheStats SlotCache::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    SlotCacheStats result;
    result.capacity_slots = slots_.size();
    result.bytes_per_slot = options_.bytes_per_slot;
    result.configured_capacity_bytes = configured_capacity_bytes_;
    result.byte_budget = options_.byte_budget;
    result.metadata_bytes = slots_.capacity() * sizeof(Slot);
    result.active_reservations = active_reservations_;
    result.active_completion_tokens = token_slots_.size();
    for (const Slot & slot : slots_) {
        if (slot.state == State::Resident) {
            ++result.resident_slots;
        } else if (slot.state == State::Reserved) {
            ++result.reserved_slots;
        }
        if (slot.gpu_use_refs != 0) {
            ++result.pinned_slots;
        }
    }
    return result;
}

}  // namespace ps5lm::moe
