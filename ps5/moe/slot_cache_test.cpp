// Host test:
//   c++ -std=c++17 -O1 -Wall -Wextra -Werror -pedantic ps5/moe/slot_cache.cpp ps5/moe/slot_cache_test.cpp -o /tmp/ps5lm_slot_cache_test
//   /tmp/ps5lm_slot_cache_test

#include "slot_cache.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

using namespace ps5lm::moe;

namespace {

constexpr ExpertKey A{0, 1};
constexpr ExpertKey B{0, 2};
constexpr ExpertKey C{1, 1};
constexpr ExpertKey D{1, 2};
constexpr ExpertKey E{2, 1};

void expect(SlotCacheStatus actual, SlotCacheStatus wanted) {
    if (actual != wanted) {
        std::cerr << "expected " << slot_cache_status_name(wanted) << ", got "
                  << slot_cache_status_name(actual) << '\n';
        std::abort();
    }
}

SlotCacheOptions options(std::size_t slots, std::size_t bytes_per_slot = 64) {
    SlotCacheOptions result;
    result.slot_count = slots;
    result.bytes_per_slot = bytes_per_slot;
    result.byte_budget = slots * bytes_per_slot;
    result.max_in_flight_tokens = 8;
    result.max_request_keys = 16;
    return result;
}

void finish_uploads(Reservation & reservation) {
    for (const SlotAssignment & entry : reservation.assignments()) {
        if (entry.needs_upload) {
            expect(reservation.mark_upload_complete(entry.key), SlotCacheStatus::Ok);
        }
    }
}

// Deliberately small independent oracle for the unpinned LRU insertion path.
// It stores only identities and timestamps and does not inspect implementation
// internals. Pinning/rollback behavior is checked separately with explicit
// adversarial cases below.
class ReferenceLru final {
public:
    explicit ReferenceLru(std::size_t count) : keys_(count), stamps_(count, 0) {}

    std::size_t touch_or_insert(ExpertKey key) {
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i].has_value() && *keys_[i] == key) {
                stamps_[i] = ++clock_;
                return i;
            }
        }
        std::size_t victim = keys_.size();
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            if (!keys_[i].has_value()) {
                victim = i;
                break;
            }
        }
        if (victim == keys_.size()) {
            victim = 0;
            for (std::size_t i = 1; i < keys_.size(); ++i) {
                if (stamps_[i] < stamps_[victim]) {
                    victim = i;
                }
            }
        }
        keys_[victim] = key;
        stamps_[victim] = ++clock_;
        return victim;
    }

private:
    std::vector<std::optional<ExpertKey>> keys_;
    std::vector<std::uint64_t> stamps_;
    std::uint64_t clock_ = 0;
};

void commit_and_release(ReserveResult & result, CompletionToken token) {
    assert(result.ok());
    finish_uploads(result.reservation);
    expect(result.reservation.commit(token), SlotCacheStatus::Ok);
    expect(result.reservation.commit(token), SlotCacheStatus::InvalidReservation);
}

void test_configuration_and_budgets() {
    SlotCache bad(options(2, 64));
    assert(bad.configuration_status() == SlotCacheStatus::Ok);
    assert(bad.stats().configured_capacity_bytes == 128);
    assert(bad.stats().capacity_slots == 2);
    assert(bad.stats().metadata_bytes >= 2 * sizeof(std::size_t));

    SlotCacheOptions too_small = options(2, 64);
    too_small.byte_budget = 127;
    SlotCache over_budget(too_small);
    expect(over_budget.configuration_status(), SlotCacheStatus::BudgetExceeded);
    assert(over_budget.slot_count() == 0);

    SlotCacheOptions overflow = options(SlotCache::kHardSlotLimit, 1);
    overflow.bytes_per_slot = std::numeric_limits<std::size_t>::max();
    overflow.byte_budget = std::numeric_limits<std::size_t>::max();
    SlotCache multiplication_overflow(overflow);
    expect(multiplication_overflow.configuration_status(), SlotCacheStatus::InvalidConfiguration);

    SlotCacheOptions invalid = options(1);
    invalid.max_request_keys = 0;
    SlotCache invalid_limits(invalid);
    expect(invalid_limits.configuration_status(), SlotCacheStatus::InvalidConfiguration);
}

void test_deduped_lru_matches_reference() {
    SlotCache cache(options(3));
    ReferenceLru oracle(3);

    auto first = cache.reserve({A, A, B});
    expect(first.status, SlotCacheStatus::Ok);
    assert(first.reservation.assignments().size() == 2);
    std::size_t slot_a = kNoSlot;
    std::size_t slot_a_repeat = kNoSlot;
    assert(first.reservation.slot_for(A, &slot_a));
    assert(first.reservation.slot_for(A, &slot_a_repeat));
    assert(slot_a == slot_a_repeat);
    assert(first.reservation.assignments()[0].key == A);
    assert(first.reservation.assignments()[1].key == B);
    assert(slot_a == oracle.touch_or_insert(A));
    std::size_t slot_b = kNoSlot;
    assert(first.reservation.slot_for(B, &slot_b));
    assert(slot_b == oracle.touch_or_insert(B));
    commit_and_release(first, 1);
    expect(cache.complete(1), SlotCacheStatus::Ok);

    auto touch_a = cache.reserve({A});
    expect(touch_a.status, SlotCacheStatus::Ok);
    std::size_t actual = kNoSlot;
    assert(touch_a.reservation.slot_for(A, &actual));
    assert(actual == oracle.touch_or_insert(A));
    commit_and_release(touch_a, 2);
    expect(cache.complete(2), SlotCacheStatus::Ok);

    auto add_c = cache.reserve({C});
    expect(add_c.status, SlotCacheStatus::Ok);
    assert(add_c.reservation.slot_for(C, &actual));
    assert(actual == oracle.touch_or_insert(C));
    commit_and_release(add_c, 3);
    expect(cache.complete(3), SlotCacheStatus::Ok);

    auto touch_b = cache.reserve({B});
    expect(touch_b.status, SlotCacheStatus::Ok);
    assert(touch_b.reservation.slot_for(B, &actual));
    assert(actual == oracle.touch_or_insert(B));
    commit_and_release(touch_b, 4);
    expect(cache.complete(4), SlotCacheStatus::Ok);

    auto add_d = cache.reserve({D});
    expect(add_d.status, SlotCacheStatus::Ok);
    assert(add_d.reservation.slot_for(D, &actual));
    assert(actual == oracle.touch_or_insert(D));
    commit_and_release(add_d, 5);
    expect(cache.complete(5), SlotCacheStatus::Ok);

    assert(!cache.find_slot(A, &actual));
    assert(cache.find_slot(B, &actual));
    assert(cache.find_slot(C, &actual));
    assert(cache.find_slot(D, &actual));
    expect(cache.complete(5), SlotCacheStatus::UnknownCompletionToken);
}

void test_capacity_failure_is_atomic_and_repeats_do_not_count() {
    SlotCache cache(options(2));
    auto too_many = cache.reserve({A, B, C});
    expect(too_many.status, SlotCacheStatus::TooManyUniqueExperts);
    assert(cache.stats().resident_slots == 0 && cache.stats().reserved_slots == 0);

    auto repeated = cache.reserve({A, A, A});
    expect(repeated.status, SlotCacheStatus::Ok);
    assert(repeated.reservation.assignments().size() == 1);
    finish_uploads(repeated.reservation);
    expect(repeated.reservation.commit(1), SlotCacheStatus::Ok);
    expect(cache.complete(1), SlotCacheStatus::Ok);
    std::size_t slot = kNoSlot;
    assert(cache.find_slot(A, &slot));

    SlotCacheOptions one_key = options(2);
    one_key.max_request_keys = 1;
    SlotCache request_limited(one_key);
    auto oversized = request_limited.reserve({A, B});
    expect(oversized.status, SlotCacheStatus::RequestTooLarge);
}

void test_simultaneous_reservations_and_pending_key_conflict() {
    SlotCache cache(options(3));
    auto pending_a = cache.reserve({A});
    auto pending_b = cache.reserve({B});
    expect(pending_a.status, SlotCacheStatus::Ok);
    expect(pending_b.status, SlotCacheStatus::Ok);
    auto conflict = cache.reserve({C, A});
    expect(conflict.status, SlotCacheStatus::KeyReserved);
    assert(cache.stats().reserved_slots == 2);
    std::size_t slot_a = kNoSlot;
    std::size_t slot_b = kNoSlot;
    assert(pending_a.reservation.slot_for(A, &slot_a));
    assert(pending_b.reservation.slot_for(B, &slot_b));
    assert(slot_a != slot_b);

    finish_uploads(pending_b.reservation);
    expect(pending_b.reservation.commit(1), SlotCacheStatus::Ok);
    finish_uploads(pending_a.reservation);
    expect(pending_a.reservation.commit(2), SlotCacheStatus::Ok);
    expect(cache.complete(1), SlotCacheStatus::Ok);
    expect(cache.complete(2), SlotCacheStatus::Ok);
    assert(cache.stats().reserved_slots == 0 && cache.stats().resident_slots == 2);
}

void test_rollback_and_partial_upload_failure() {
    SlotCache cache(options(2));
    auto load_a = cache.reserve({A});
    expect(load_a.status, SlotCacheStatus::Ok);
    finish_uploads(load_a.reservation);
    expect(load_a.reservation.commit(1), SlotCacheStatus::Ok);
    expect(cache.complete(1), SlotCacheStatus::Ok);

    std::size_t a_slot = kNoSlot;
    assert(cache.find_slot(A, &a_slot));
    auto batch = cache.reserve({A, B, C});
    expect(batch.status, SlotCacheStatus::TooManyUniqueExperts);
    assert(cache.find_slot(A, &a_slot));
    assert(cache.stats().resident_slots == 1 && cache.stats().reserved_slots == 0);

    auto failed = cache.reserve({A, B});
    expect(failed.status, SlotCacheStatus::Ok);
    expect(failed.reservation.mark_upload_complete(A), SlotCacheStatus::UploadNotRequired);
    assert(failed.reservation.mark_upload_complete(B) == SlotCacheStatus::Ok);
    assert(failed.reservation.mark_upload_complete(C) == SlotCacheStatus::InvalidRequest);
    expect(failed.reservation.commit(2), SlotCacheStatus::Ok);
    expect(cache.complete(2), SlotCacheStatus::Ok);

    auto partial = cache.reserve({B, C});
    expect(partial.status, SlotCacheStatus::Ok);
    finish_uploads(partial.reservation); // caller can roll back after a later upload failure
    partial.reservation.rollback();
    assert(!partial.reservation.active());
    assert(!cache.find_slot(C, &a_slot));
    assert(cache.find_slot(B, &a_slot));
    assert(cache.stats().reserved_slots == 0);
    auto empty_reuse = cache.reserve({C});
    expect(empty_reuse.status, SlotCacheStatus::Ok);
    finish_uploads(empty_reuse.reservation);
    expect(empty_reuse.reservation.commit(3), SlotCacheStatus::Ok);
    expect(cache.complete(3), SlotCacheStatus::Ok);

    auto incomplete = cache.reserve({E});
    expect(incomplete.status, SlotCacheStatus::Ok);
    expect(incomplete.reservation.commit(4), SlotCacheStatus::UploadsIncomplete);
    incomplete.reservation.rollback();
    std::size_t slot_e = kNoSlot;
    assert(!cache.find_slot(E, &slot_e));
    // The incomplete upload never becomes visible and rollback frees its slot.
    assert(cache.stats().reserved_slots == 0);
}

void test_gpu_pins_protect_lru_until_every_token_completes() {
    SlotCache cache(options(2));
    auto load_a = cache.reserve({A});
    expect(load_a.status, SlotCacheStatus::Ok);
    finish_uploads(load_a.reservation);
    expect(load_a.reservation.commit(1), SlotCacheStatus::Ok);

    auto second_use = cache.reserve({A});
    expect(second_use.status, SlotCacheStatus::Ok);
    expect(second_use.reservation.commit(2), SlotCacheStatus::Ok);

    auto load_b = cache.reserve({B});
    expect(load_b.status, SlotCacheStatus::Ok);
    finish_uploads(load_b.reservation);
    expect(load_b.reservation.commit(3), SlotCacheStatus::Ok);
    std::size_t slot = kNoSlot;
    assert(cache.find_slot(A, &slot));
    expect(cache.complete(1), SlotCacheStatus::Ok);
    assert(cache.find_slot(A, &slot));

    auto no_evict = cache.reserve({C});
    expect(no_evict.status, SlotCacheStatus::InsufficientUnpinnedSlots);
    assert(cache.find_slot(A, &slot));
    assert(cache.find_slot(B, &slot));
    assert(cache.stats().resident_slots == 2 && cache.stats().reserved_slots == 0);

    expect(cache.complete(2), SlotCacheStatus::Ok);
    auto evict_a = cache.reserve({C});
    expect(evict_a.status, SlotCacheStatus::Ok);
    assert(!cache.find_slot(A, &slot));
    assert(cache.find_slot(B, &slot));
    finish_uploads(evict_a.reservation);
    expect(evict_a.reservation.commit(4), SlotCacheStatus::Ok);
    expect(cache.complete(3), SlotCacheStatus::Ok);
    expect(cache.complete(4), SlotCacheStatus::Ok);
}

void test_failed_read_rollback_and_token_limits() {
    SlotCacheOptions opts = options(1);
    opts.max_in_flight_tokens = 1;
    SlotCache cache(opts);
    auto first = cache.reserve({A});
    expect(first.status, SlotCacheStatus::Ok);
    finish_uploads(first.reservation);
    expect(first.reservation.commit(4), SlotCacheStatus::Ok);

    auto second = cache.reserve({A});
    expect(second.status, SlotCacheStatus::Ok);
    expect(second.reservation.commit(5), SlotCacheStatus::TooManyInFlightTokens);
    assert(second.reservation.active());
    second.reservation.rollback();
    expect(cache.complete(4), SlotCacheStatus::Ok);

    auto failed_read = cache.reserve({B});
    expect(failed_read.status, SlotCacheStatus::Ok);
    // Simulate a read error before upload: no publish and automatic rollback.
    failed_read.reservation.rollback();
    std::size_t slot = kNoSlot;
    assert(!cache.find_slot(B, &slot));
    assert(!cache.find_slot(A, &slot)); // reserved victim is conservatively invalidated

    auto cannot_reuse_old_token = cache.reserve({B});
    expect(cannot_reuse_old_token.status, SlotCacheStatus::Ok);
    finish_uploads(cannot_reuse_old_token.reservation);
    expect(cannot_reuse_old_token.reservation.commit(4), SlotCacheStatus::InvalidCompletionToken);
    expect(cannot_reuse_old_token.reservation.commit(3), SlotCacheStatus::InvalidCompletionToken);
    cannot_reuse_old_token.reservation.rollback();
    assert(!cache.find_slot(B, &slot));
}

void test_cancellation_raii_and_reservation_pins() {
    SlotCache cache(options(1));
    {
        auto cancelled = cache.reserve({A});
        expect(cancelled.status, SlotCacheStatus::Ok);
        // Cancellation after partial CPU work abandons the unpublished slot.
        assert(cancelled.reservation.mark_upload_complete(A) == SlotCacheStatus::Ok);
    } // RAII rollback is the cancellation path if the caller exits early.
    assert(cache.stats().reserved_slots == 0);
    std::size_t slot = kNoSlot;
    assert(!cache.find_slot(A, &slot));

    auto load_a = cache.reserve({A});
    expect(load_a.status, SlotCacheStatus::Ok);
    finish_uploads(load_a.reservation);
    expect(load_a.reservation.commit(1), SlotCacheStatus::Ok);
    expect(cache.complete(1), SlotCacheStatus::Ok);

    auto hold_a = cache.reserve({A});
    expect(hold_a.status, SlotCacheStatus::Ok);
    auto cannot_evict_reserved_hit = cache.reserve({B});
    expect(cannot_evict_reserved_hit.status, SlotCacheStatus::InsufficientUnpinnedSlots);
    hold_a.reservation.rollback();

    auto can_evict_after_cancel = cache.reserve({B});
    expect(can_evict_after_cancel.status, SlotCacheStatus::Ok);
    assert(!cache.find_slot(A, &slot));
    finish_uploads(can_evict_after_cancel.reservation);
    expect(can_evict_after_cancel.reservation.commit(2), SlotCacheStatus::Ok);
    expect(cache.complete(2), SlotCacheStatus::Ok);
}

}  // namespace

int main() {
    test_configuration_and_budgets();
    test_deduped_lru_matches_reference();
    test_capacity_failure_is_atomic_and_repeats_do_not_count();
    test_simultaneous_reservations_and_pending_key_conflict();
    test_rollback_and_partial_upload_failure();
    test_gpu_pins_protect_lru_until_every_token_completes();
    test_failed_read_rollback_and_token_limits();
    test_cancellation_raii_and_reservation_pins();
    std::cout << "slot_cache_test: PASS\n";
}
