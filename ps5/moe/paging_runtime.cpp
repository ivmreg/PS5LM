// Synchronous per-layer adapter between ExpertSource and SlotCache.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "paging_runtime.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
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

PagingQuarantine::PagingQuarantine(
        std::size_t staging_capacity,
        std::size_t upload_batch_capacity) {
    if (staging_capacity == 0 || upload_batch_capacity == 0 ||
        upload_batch_capacity > kHardMaxUploadBatchCapacity ||
        staging_capacity > std::numeric_limits<std::size_t>::max() / upload_batch_capacity) {
        return;
    }
    staging_capacity_ = staging_capacity;
    upload_batch_capacity_ = upload_batch_capacity;
    staging_.resize(staging_capacity * upload_batch_capacity);
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
        const QuarantinedTransferInfo * transfers,
        std::size_t transfer_count) noexcept {
    if (pending_ || !reservation.active() || !leased_ || backend_ == nullptr ||
        transfers == nullptr || transfer_count == 0 ||
        transfer_count > pending_transfers_.size()) {
        std::terminate();
    }
    pending_.emplace(std::move(reservation));
    std::copy_n(transfers, transfer_count, pending_transfers_.begin());
    pending_transfer_count_ = transfer_count;
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
    pending_transfers_ = {};
    pending_transfer_count_ = 0;
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
        quarantine.upload_batch_capacity() == 0 ||
        quarantine.upload_batch_capacity() > kHardMaxUploadBatchCapacity ||
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

    struct PendingUpload {
        QuarantinedTransferInfo transfer;
        std::size_t assignment_index = 0;
        bool has_fence_token = false;
    };
    enum class DrainOutcome : std::uint8_t { Complete, UploadFailed, CacheFailed, Unresolved };

    std::array<PendingUpload, kHardMaxUploadBatchCapacity> pending{};
    std::array<QuarantinedTransferInfo, kHardMaxUploadBatchCapacity> unresolved{};
    std::array<UploadToken, kHardMaxUploadBatchCapacity> wait_tokens{};
    std::array<TransferFenceStatus, kHardMaxUploadBatchCapacity> wait_statuses{};
    std::array<std::uint8_t, kHardSelectedExpertLimit> completed_roles{};
    std::size_t pending_count = 0;
    std::size_t unresolved_count = 0;
    bool saw_upload_failure = false;
    SlotCacheStatus drain_cache_failure = SlotCacheStatus::Ok;

    const auto drain_pending = [&]() noexcept -> DrainOutcome {
        std::size_t wait_count = 0;
        for (std::size_t i = 0; i < pending_count; ++i) {
            if (pending[i].has_fence_token) {
                wait_tokens[wait_count++] = pending[i].transfer.token;
            }
        }
        wait_statuses.fill(TransferFenceStatus::Indeterminate);
        if (wait_count != 0) {
            backend_.wait_transfers(wait_tokens.data(), wait_count, wait_statuses.data());
        }

        std::size_t wait_index = 0;
        for (std::size_t i = 0; i < pending_count; ++i) {
            const PendingUpload & upload = pending[i];
            if (!upload.has_fence_token) {
                unresolved[unresolved_count++] = upload.transfer;
                continue;
            }

            const TransferFenceStatus status = wait_statuses[wait_index++];
            if (status == TransferFenceStatus::Indeterminate) {
                unresolved[unresolved_count++] = upload.transfer;
                continue;
            }
            if (status == TransferFenceStatus::FailedButQuiesced) {
                if (!saw_upload_failure) {
                    result.failed_key = upload.transfer.key;
                    result.failed_role = upload.transfer.role;
                    result.transfer_token = upload.transfer.token;
                }
                saw_upload_failure = true;
                continue;
            }

            std::uint8_t & role_count = completed_roles[upload.assignment_index];
            ++role_count;
            if (role_count == roles.size()) {
                const SlotAssignment & assignment = reservation.assignments()[upload.assignment_index];
                const SlotCacheStatus marked = reservation.mark_upload_complete(assignment.key);
                if (marked != SlotCacheStatus::Ok && drain_cache_failure == SlotCacheStatus::Ok) {
                    drain_cache_failure = marked;
                    result.failed_key = assignment.key;
                    result.failed_role = upload.transfer.role;
                }
            }
        }
        pending_count = 0;

        if (unresolved_count != 0) {
            result.status = PagingStatus::FenceUnresolved;
            result.transfer_token = unresolved[0].token;
            result.failed_key = unresolved[0].key;
            result.failed_role = unresolved[0].role;
            quarantine_.quarantine(std::move(reservation), unresolved.data(), unresolved_count);
            poisoned_ = true;
            return DrainOutcome::Unresolved;
        }
        if (drain_cache_failure != SlotCacheStatus::Ok) {
            result.status = PagingStatus::CacheFailure;
            result.cache_status = drain_cache_failure;
            return DrainOutcome::CacheFailed;
        }
        if (saw_upload_failure) {
            result.status = PagingStatus::UploadFailed;
            return DrainOutcome::UploadFailed;
        }
        return DrainOutcome::Complete;
    };

    const auto abort_layer = [&](PagingStatus cause) noexcept -> PagingResult {
        const DrainOutcome drained = drain_pending();
        if (drained == DrainOutcome::Complete) {
            reservation.rollback();
            result.status = cause;
        } else if (reservation.active()) {
            reservation.rollback();
        }
        return std::move(result);
    };

    const auto & assignments = reservation.assignments();
    for (std::size_t assignment_index = 0; assignment_index < assignments.size(); ++assignment_index) {
        const SlotAssignment & assignment = assignments[assignment_index];
        if (!assignment.needs_upload) {
            continue;
        }
        for (const auto & role_ref : roles) {
            const ExpertTensorRole role = role_ref.first;
            const ExpertTensorRef & tensor = role_ref.second;
            result.failed_key = assignment.key;
            result.failed_role = role;

            if (is_cancelled(cancelled)) {
                return abort_layer(PagingStatus::Cancelled);
            }

            const std::size_t staging_index = pending_count;
            std::uint8_t * const staging_bytes = quarantine_.staging_.data() +
                    staging_index * quarantine_.staging_capacity_;
            const ReadError read = source_.read_expert(
                    tensor.tensor_id,
                    assignment.key.expert,
                    staging_bytes,
                    tensor.bytes_per_expert,
                    cancelled);
            if (!read.ok()) {
                result.read_error = read;
                return abort_layer(read.status == ReadStatus::Cancelled ?
                        PagingStatus::Cancelled : PagingStatus::SourceFailure);
            }

            if (is_cancelled(cancelled)) {
                return abort_layer(PagingStatus::Cancelled);
            }

            const UploadSubmission submission = backend_.submit_upload(
                    assignment.key,
                    role,
                    assignment.slot,
                    staging_bytes,
                    tensor.bytes_per_expert);
            if (!submission.submitted) {
                return abort_layer(PagingStatus::UploadRejected);
            }

            result.transfer_token = submission.token;
            pending[pending_count++] = {
                {submission.token, assignment.key, role, assignment.slot},
                assignment_index,
                submission.token != 0,
            };
            if (submission.token == 0) {
                const DrainOutcome drained = drain_pending();
                if (drained != DrainOutcome::Complete && reservation.active()) {
                    reservation.rollback();
                }
                if (drained == DrainOutcome::Complete) {
                    // A submitted operation without a token cannot be fenced individually.
                    // If drain_pending did not quarantine it, quarantine explicitly and poison the runtime.
                    const QuarantinedTransferInfo zero_token_info{
                        submission.token, assignment.key, role, assignment.slot
                    };
                    quarantine_.quarantine(std::move(reservation), &zero_token_info, 1);
                    poisoned_ = true;
                    result.status = PagingStatus::FenceUnresolved;
                    result.transfer_token = submission.token;
                    result.failed_key = assignment.key;
                    result.failed_role = role;
                }
                return result;
            }

            if (pending_count == quarantine_.upload_batch_capacity()) {
                const DrainOutcome drained = drain_pending();
                if (drained != DrainOutcome::Complete) {
                    if (reservation.active()) {
                        reservation.rollback();
                    }
                    return result;
                }
            }
        }
    }

    if (pending_count != 0) {
        const DrainOutcome drained = drain_pending();
        if (drained != DrainOutcome::Complete) {
            if (reservation.active()) {
                reservation.rollback();
            }
            return result;
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
