// Synchronous per-layer adapter between ExpertSource and SlotCache.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "paging_runtime.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <new>
#include <utility>

namespace ps5lm::moe {

const char * paging_status_name(PagingStatus status) noexcept {
    switch (status) {
        case PagingStatus::Ok: return "ok";
        case PagingStatus::InvalidConfiguration: return "invalid_configuration";
        case PagingStatus::InvalidLayer: return "invalid_layer";
        case PagingStatus::RequestTooLarge: return "request_too_large";
        case PagingStatus::Cancelled: return "cancelled";
        case PagingStatus::CacheFailure: return "cache_failure";
        case PagingStatus::SourceFailure: return "source_failure";
        case PagingStatus::UploadRejected: return "upload_rejected";
        case PagingStatus::UploadFailed: return "upload_failed";
        case PagingStatus::FenceUnresolved: return "fence_unresolved";
        case PagingStatus::Quarantined: return "quarantined";
        case PagingStatus::OutOfMemory: return "out_of_memory";
    }
    return "unknown";
}

PagingQuarantine::PagingQuarantine(std::size_t staging_capacity) {
    if (staging_capacity != 0) {
        staging_.resize(staging_capacity);
    }
}

PagingQuarantine::~PagingQuarantine() noexcept {
    // Dropping an unresolved Reservation would invoke its rollback destructor,
    // potentially making a slot available while a transfer still uses it.
    if (leased_ || pending_) {
        std::terminate();
    }
}

bool PagingQuarantine::acquire(ExpertUploadBackend & backend) noexcept {
    if (leased_ || pending_ || staging_.empty() || (backend_ != nullptr && backend_ != &backend)) {
        return false;
    }
    leased_ = true;
    backend_ = &backend;
    return true;
}

void PagingQuarantine::release() noexcept {
    leased_ = false;
}

void PagingQuarantine::quarantine(
        Reservation && reservation,
        ExpertKey key,
        ExpertTensorRole role,
        std::size_t slot,
        UploadToken transfer) noexcept {
    if (pending_ || !reservation.active() || !leased_ || backend_ == nullptr) {
        std::terminate();
    }
    pending_.emplace(std::move(reservation));
    pending_transfer_ = transfer;
    pending_key_ = key;
    pending_role_ = role;
    pending_slot_ = slot;
}

QuarantineStatus PagingQuarantine::release_after_backend_quiesced(ExpertUploadBackend & backend) noexcept {
    if (!pending_) {
        return QuarantineStatus::NotPending;
    }
    if (backend_ != &backend) {
        return QuarantineStatus::StillUnsafe;
    }
    const TransferFenceStatus status = backend.quiesce();
    if (status == TransferFenceStatus::Indeterminate) {
        return QuarantineStatus::StillUnsafe;
    }
    pending_->rollback();
    pending_.reset();
    pending_transfer_ = 0;
    pending_key_ = {};
    pending_role_ = ExpertTensorRole::Gate;
    pending_slot_ = kNoSlot;
    if (!leased_) {
        backend_ = nullptr;
    }
    return QuarantineStatus::Released;
}

PagingRuntime::PagingRuntime(
        SlotCache & cache,
        const ExpertSource & source,
        ExpertUploadBackend & backend,
        PagingQuarantine & quarantine,
        const std::vector<LayerExpertTensorIds> & tensor_ids,
        std::size_t max_selected_experts)
    : cache_(cache),
      source_(source),
      backend_(backend),
      quarantine_(quarantine),
      max_selected_experts_(max_selected_experts) {
    if (cache.configuration_status() != SlotCacheStatus::Ok ||
        source.max_expert_bytes() == 0 || quarantine.staging_capacity() == 0 ||
        quarantine.staging_capacity() > source.max_expert_bytes() ||
        max_selected_experts == 0 || max_selected_experts > kHardSelectedExpertLimit ||
        tensor_ids.empty() || tensor_ids.size() > kHardLayerLimit) {
        configuration_status_ = PagingStatus::InvalidConfiguration;
        return;
    }

    try {
        for (const LayerExpertTensorIds & layer : tensor_ids) {
            const std::array<ExpertTensorRef, 3> refs = {layer.gate, layer.up, layer.down};
            if (refs[0].tensor_id == refs[1].tensor_id || refs[0].tensor_id == refs[2].tensor_id ||
                refs[1].tensor_id == refs[2].tensor_id) {
                configuration_status_ = PagingStatus::InvalidConfiguration;
                return;
            }
            for (const ExpertTensorRef & ref : refs) {
                if (ref.bytes_per_expert == 0 || ref.bytes_per_expert > quarantine.staging_capacity()) {
                    configuration_status_ = PagingStatus::InvalidConfiguration;
                    return;
                }
            }
            if (!tensor_ids_.emplace(layer.layer, layer).second) {
                configuration_status_ = PagingStatus::InvalidConfiguration;
                return;
            }
        }
    } catch (const std::bad_alloc &) {
        configuration_status_ = PagingStatus::OutOfMemory;
        return;
    } catch (...) {
        configuration_status_ = PagingStatus::OutOfMemory;
        return;
    }

    if (!quarantine.acquire(backend)) {
        configuration_status_ = PagingStatus::InvalidConfiguration;
        return;
    }
    leased_ = true;
    configuration_status_ = PagingStatus::Ok;
}

PagingRuntime::~PagingRuntime() noexcept {
    if (leased_) {
        quarantine_.release();
    }
}

const LayerExpertTensorIds * PagingRuntime::find_layer(std::uint32_t layer) const noexcept {
    const auto found = tensor_ids_.find(layer);
    return found == tensor_ids_.end() ? nullptr : &found->second;
}

bool PagingRuntime::is_cancelled(const std::atomic_bool * cancelled) const noexcept {
    return cancelled != nullptr && cancelled->load(std::memory_order_relaxed);
}

PagingResult PagingRuntime::fail_with_cache(PagingResult result, SlotCacheStatus status) const noexcept {
    result.status = PagingStatus::CacheFailure;
    result.cache_status = status;
    return result;
}

PagingResult PagingRuntime::load_layer(
        std::uint32_t layer,
        const std::vector<std::uint32_t> & selected_experts,
        CompletionToken gpu_use_token,
        const std::atomic_bool * cancelled) noexcept {
    PagingResult result;
    if (configuration_status_ != PagingStatus::Ok) {
        result.status = configuration_status_;
        return result;
    }
    if (poisoned_ || quarantine_.pending()) {
        result.status = PagingStatus::Quarantined;
        return result;
    }
    if (is_cancelled(cancelled)) {
        result.status = PagingStatus::Cancelled;
        return result;
    }
    if (selected_experts.empty()) {
        return fail_with_cache(std::move(result), SlotCacheStatus::InvalidRequest);
    }
    if (selected_experts.size() > max_selected_experts_) {
        result.status = PagingStatus::RequestTooLarge;
        return result;
    }
    const LayerExpertTensorIds * tensors = find_layer(layer);
    if (tensors == nullptr) {
        result.status = PagingStatus::InvalidLayer;
        return result;
    }

    std::vector<ExpertKey> keys;
    try {
        keys.reserve(selected_experts.size());
        for (const std::uint32_t expert : selected_experts) {
            keys.push_back({layer, expert});
        }
    } catch (...) {
        result.status = PagingStatus::OutOfMemory;
        return result;
    }

    ReserveResult reserved;
    try {
        reserved = cache_.reserve(keys);
    } catch (...) {
        result.status = PagingStatus::OutOfMemory;
        return result;
    }
    if (!reserved.ok()) {
        return fail_with_cache(std::move(result), reserved.status);
    }
    Reservation & reservation = reserved.reservation;

    const std::array<std::pair<ExpertTensorRole, ExpertTensorRef>, 3> roles = {{
        {ExpertTensorRole::Gate, tensors->gate},
        {ExpertTensorRole::Up,   tensors->up},
        {ExpertTensorRole::Down, tensors->down},
    }};

    for (const SlotAssignment & assignment : reservation.assignments()) {
        if (!assignment.needs_upload) {
            continue;
        }
        for (const auto & role_ref : roles) {
            const ExpertTensorRole role = role_ref.first;
            const ExpertTensorRef & tensor = role_ref.second;
            result.failed_key = assignment.key;
            result.failed_role = role;

            if (is_cancelled(cancelled)) {
                reservation.rollback();
                result.status = PagingStatus::Cancelled;
                return result;
            }

            std::fill_n(quarantine_.staging_.begin(), tensor.bytes_per_expert, 0);
            const ReadError read = source_.read_expert(
                    tensor.tensor_id,
                    assignment.key.expert,
                    quarantine_.staging_.data(),
                    tensor.bytes_per_expert,
                    cancelled);
            if (!read.ok()) {
                reservation.rollback();
                result.read_error = read;
                result.status = read.status == ReadStatus::Cancelled ?
                        PagingStatus::Cancelled : PagingStatus::SourceFailure;
                return result;
            }

            if (is_cancelled(cancelled)) {
                reservation.rollback();
                result.status = PagingStatus::Cancelled;
                return result;
            }

            const UploadSubmission submission = backend_.submit_upload(
                    assignment.key,
                    role,
                    assignment.slot,
                    quarantine_.staging_.data(),
                    tensor.bytes_per_expert);
            if (!submission.submitted) {
                reservation.rollback();
                result.status = PagingStatus::UploadRejected;
                return result;
            }

            result.transfer_token = submission.token;
            if (submission.token == 0) {
                quarantine_.quarantine(
                        std::move(reservation), assignment.key, role, assignment.slot, submission.token);
                poisoned_ = true;
                result.status = PagingStatus::FenceUnresolved;
                return result;
            }

            const TransferFenceStatus transfer_status = backend_.wait_transfer(submission.token);
            if (transfer_status == TransferFenceStatus::Indeterminate) {
                quarantine_.quarantine(
                        std::move(reservation), assignment.key, role, assignment.slot, submission.token);
                poisoned_ = true;
                result.status = PagingStatus::FenceUnresolved;
                return result;
            }
            if (transfer_status == TransferFenceStatus::FailedButQuiesced) {
                reservation.rollback();
                result.status = PagingStatus::UploadFailed;
                return result;
            }

            if (is_cancelled(cancelled)) {
                reservation.rollback();
                result.status = PagingStatus::Cancelled;
                return result;
            }
        }

        const SlotCacheStatus marked = reservation.mark_upload_complete(assignment.key);
        if (marked != SlotCacheStatus::Ok) {
            reservation.rollback();
            return fail_with_cache(std::move(result), marked);
        }
    }

    if (is_cancelled(cancelled)) {
        reservation.rollback();
        result.status = PagingStatus::Cancelled;
        return result;
    }

    try {
        result.assignments = reservation.assignments();
    } catch (...) {
        reservation.rollback();
        result.status = PagingStatus::OutOfMemory;
        return result;
    }
    const SlotCacheStatus committed = reservation.commit(gpu_use_token);
    if (committed != SlotCacheStatus::Ok) {
        reservation.rollback();
        result.assignments.clear();
        return fail_with_cache(std::move(result), committed);
    }
    result.gpu_use_token = gpu_use_token;
    result.status = PagingStatus::Ok;
    return result;
}

}  // namespace ps5lm::moe
