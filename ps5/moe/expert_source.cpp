#include "expert_source.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ps5lm::moe {
namespace {

struct FileHandle {
    int value = -1;

    FileHandle() noexcept = default;
    explicit FileHandle(int fd) noexcept : value(fd) {}
    ~FileHandle() noexcept {
        if (value >= 0) {
            ::close(value);
        }
    }

    FileHandle(const FileHandle &) = delete;
    FileHandle & operator=(const FileHandle &) = delete;

    FileHandle(FileHandle && other) noexcept : value(other.value) {
        other.value = -1;
    }

    FileHandle & operator=(FileHandle && other) noexcept {
        if (this != &other) {
            if (value >= 0) {
                ::close(value);
            }
            value = other.value;
            other.value = -1;
        }
        return *this;
    }
};

ReadError make_error(ReadStatus status, int system_error, const char * message) noexcept {
    ReadError error;
    error.status = status;
    error.system_error = system_error;
    std::snprintf(error.message, sizeof(error.message), "%s", message ? message : "error");
    return error;
}

OpenResult open_failure(ReadStatus status, int system_error, const char * message) noexcept {
    OpenResult result;
    result.error = make_error(status, system_error, message);
    return result;
}

bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t & out) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t & out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool fits_off_t(std::uint64_t value) noexcept {
    return value <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
}

bool identity_from_stat(const struct stat & st, FileIdentity & identity) noexcept {
    identity.device = static_cast<std::uint64_t>(st.st_dev);
    identity.inode = static_cast<std::uint64_t>(st.st_ino);
    identity.available = st.st_ino != 0;
    return identity.available;
}

bool same_identity(const FileIdentity & a, const FileIdentity & b) noexcept {
    return a.available && b.available && a.device == b.device && a.inode == b.inode;
}

ReadError validate_current_file(
        int fd,
        const char * path,
        std::uint64_t expected_size,
        const FileIdentity & opened_identity) noexcept {
    struct stat fd_stat{};
    if (::fstat(fd, &fd_stat) != 0) {
        return make_error(ReadStatus::IoError, errno, "fstat failed for expert shard");
    }
    if (fd_stat.st_size < 0 || static_cast<std::uint64_t>(fd_stat.st_size) != expected_size) {
        return make_error(ReadStatus::FileChanged, 0, "expert shard size changed");
    }

    FileIdentity current_fd_identity;
    if (identity_from_stat(fd_stat, current_fd_identity) && opened_identity.available &&
            !same_identity(current_fd_identity, opened_identity)) {
        return make_error(ReadStatus::FileChanged, 0, "open expert shard identity changed");
    }

    if (opened_identity.available) {
        struct stat path_stat{};
        if (::stat(path, &path_stat) != 0) {
            return make_error(ReadStatus::FileChanged, errno, "expert shard path is no longer available");
        }
        FileIdentity current_path_identity;
        if (!identity_from_stat(path_stat, current_path_identity) ||
                !same_identity(current_path_identity, opened_identity)) {
            return make_error(ReadStatus::FileChanged, 0, "expert shard path now names a different file");
        }
        if (path_stat.st_size < 0 || static_cast<std::uint64_t>(path_stat.st_size) != expected_size) {
            return make_error(ReadStatus::FileChanged, 0, "expert shard path size changed");
        }
    }

    return {};
}

}  // namespace

struct ExpertSource::Shard {
    std::string path;
    std::uint64_t expected_size = 0;
    FileIdentity expected_identity{};
    FileIdentity opened_identity{};
    FileHandle handle;
};

ExpertSource::ExpertSource() noexcept = default;
ExpertSource::~ExpertSource() noexcept = default;

OpenResult ExpertSource::open(
        const std::vector<ShardPath> & shard_paths,
        const std::vector<ExpertTensorDescriptor> & tensor_descriptors,
        OpenOptions options) noexcept {
    if (shard_paths.empty() || tensor_descriptors.empty() || options.max_expert_bytes == 0) {
        return open_failure(ReadStatus::InvalidArgument, 0, "shards, tensor descriptors, and a nonzero byte cap are required");
    }
    if (shard_paths.size() > kMaxShards) {
        return open_failure(ReadStatus::TooManyShards, 0, "shard count exceeds the source limit");
    }
    if (tensor_descriptors.size() > kMaxTensorDescriptors) {
        return open_failure(ReadStatus::TooManyTensors, 0, "tensor descriptor count exceeds the source limit");
    }

    try {
        auto source = std::unique_ptr<ExpertSource>(new ExpertSource());
        source->max_expert_bytes_ = options.max_expert_bytes;
        source->shards_.resize(shard_paths.size());
        source->tensors_ = tensor_descriptors;

        for (std::size_t i = 0; i < shard_paths.size(); ++i) {
            if (shard_paths[i].path.empty() || shard_paths[i].path.size() > kMaxPathBytes ||
                    shard_paths[i].path.find('\0') != std::string::npos) {
                return open_failure(ReadStatus::InvalidArgument, 0, "shard path is empty or exceeds the path limit");
            }
            source->shards_[i].path = shard_paths[i].path;
        }

        std::vector<bool> shard_has_descriptor(shard_paths.size(), false);
        for (const ExpertTensorDescriptor & tensor : source->tensors_) {
            if (tensor.shard_index >= source->shards_.size() || tensor.tensor_bytes == 0 ||
                    tensor.expert_count == 0 || tensor.expert_stride_bytes == 0 ||
                    tensor.expected_file_size == 0) {
                return open_failure(ReadStatus::InvalidDescriptor, 0, "expert tensor descriptor has a zero or invalid field");
            }

            if (tensor.expert_stride_bytes > options.max_expert_bytes ||
                    tensor.expert_stride_bytes > std::numeric_limits<std::size_t>::max()) {
                return open_failure(ReadStatus::BufferLimitExceeded, 0, "expert slice exceeds the configured staging cap");
            }

            std::uint64_t expected_tensor_bytes = 0;
            std::uint64_t tensor_end = 0;
            if (!checked_mul(tensor.expert_count, tensor.expert_stride_bytes, expected_tensor_bytes) ||
                    expected_tensor_bytes != tensor.tensor_bytes ||
                    !checked_add(tensor.absolute_offset, tensor.tensor_bytes, tensor_end) ||
                    tensor_end > tensor.expected_file_size ||
                    !fits_off_t(tensor.absolute_offset) || !fits_off_t(tensor_end) ||
                    !fits_off_t(tensor.expected_file_size)) {
                return open_failure(ReadStatus::InvalidDescriptor, 0, "expert tensor range or expert stride is invalid");
            }

            Shard & shard = source->shards_[tensor.shard_index];
            if (!shard_has_descriptor[tensor.shard_index]) {
                shard.expected_size = tensor.expected_file_size;
                shard.expected_identity = tensor.expected_identity;
                shard_has_descriptor[tensor.shard_index] = true;
            } else if (shard.expected_size != tensor.expected_file_size ||
                    shard.expected_identity.available != tensor.expected_identity.available ||
                    (shard.expected_identity.available &&
                     (shard.expected_identity.device != tensor.expected_identity.device ||
                      shard.expected_identity.inode != tensor.expected_identity.inode))) {
                return open_failure(ReadStatus::InvalidDescriptor, 0, "descriptors disagree about their shard identity or size");
            }
        }

        if (std::find(shard_has_descriptor.begin(), shard_has_descriptor.end(), false) != shard_has_descriptor.end()) {
            return open_failure(ReadStatus::InvalidDescriptor, 0, "every shard path must have a tensor descriptor");
        }

        std::sort(source->tensors_.begin(), source->tensors_.end(), [](const auto & a, const auto & b) {
            return a.tensor_id < b.tensor_id;
        });
        for (std::size_t i = 1; i < source->tensors_.size(); ++i) {
            if (source->tensors_[i - 1].tensor_id == source->tensors_[i].tensor_id) {
                return open_failure(ReadStatus::InvalidDescriptor, 0, "tensor identifiers must be unique");
            }
        }

#ifdef PS5LM_MOE_EXPERT_SOURCE_TESTING
        source->pread_for_test_ = options.pread_for_test;
        source->pread_user_data_ = options.pread_user_data;
#endif

        for (Shard & shard : source->shards_) {
            // Avoid blocking on a FIFO before fstat can reject it. O_NONBLOCK
            // has no effect on regular-file reads.
            const int fd = ::open(shard.path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) {
                return open_failure(ReadStatus::FileOpenFailed, errno, "could not open expert shard read-only");
            }
            shard.handle = FileHandle(fd);

            struct stat fd_stat{};
            if (::fstat(fd, &fd_stat) != 0) {
                return open_failure(ReadStatus::IoError, errno, "fstat failed while opening expert shard");
            }
            if (!S_ISREG(fd_stat.st_mode)) {
                return open_failure(ReadStatus::InvalidDescriptor, 0, "expert shard is not a regular file");
            }
            if (fd_stat.st_size < 0 || static_cast<std::uint64_t>(fd_stat.st_size) != shard.expected_size) {
                return open_failure(ReadStatus::FileChanged, 0, "expert shard length does not match its descriptor");
            }
            identity_from_stat(fd_stat, shard.opened_identity);
            if (shard.expected_identity.available &&
                    !same_identity(shard.expected_identity, shard.opened_identity)) {
                return open_failure(ReadStatus::FileIdentityMismatch, 0, "expert shard identity does not match its descriptor");
            }

            struct stat path_stat{};
            if (::stat(shard.path.c_str(), &path_stat) != 0) {
                return open_failure(ReadStatus::IoError, errno, "stat failed while opening expert shard");
            }
            if (path_stat.st_size < 0 || static_cast<std::uint64_t>(path_stat.st_size) != shard.expected_size) {
                return open_failure(ReadStatus::FileChanged, 0, "expert shard path length does not match its descriptor");
            }
            FileIdentity path_identity;
            if (identity_from_stat(path_stat, path_identity) && shard.opened_identity.available &&
                    !same_identity(path_identity, shard.opened_identity)) {
                return open_failure(ReadStatus::FileIdentityMismatch, 0, "expert shard path changed during open");
            }
        }

        OpenResult result;
        result.source = std::move(source);
        return result;
    } catch (const std::bad_alloc &) {
        return open_failure(ReadStatus::OutOfMemory, 0, "could not allocate bounded expert source metadata");
    } catch (...) {
        return open_failure(ReadStatus::InvalidArgument, 0, "unexpected error while opening expert source");
    }
}

ReadError ExpertSource::read_expert(
        std::uint64_t tensor_id,
        std::uint64_t expert_id,
        void * destination,
        std::size_t destination_capacity,
        const std::atomic_bool * cancelled) const noexcept {
    if (destination == nullptr) {
        return make_error(ReadStatus::InvalidArgument, 0, "destination buffer is null");
    }
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
        return make_error(ReadStatus::Cancelled, 0, "expert read was cancelled");
    }

    const auto it = std::lower_bound(tensors_.begin(), tensors_.end(), tensor_id,
            [](const ExpertTensorDescriptor & tensor, std::uint64_t id) {
                return tensor.tensor_id < id;
            });
    if (it == tensors_.end() || it->tensor_id != tensor_id) {
        return make_error(ReadStatus::TensorNotFound, 0, "expert tensor identifier was not found");
    }
    if (expert_id >= it->expert_count) {
        return make_error(ReadStatus::ExpertOutOfRange, 0, "expert index is outside the tensor");
    }
    if (it->expert_stride_bytes > max_expert_bytes_ ||
            it->expert_stride_bytes > std::numeric_limits<std::size_t>::max()) {
        return make_error(ReadStatus::BufferLimitExceeded, 0, "expert slice exceeds the configured staging cap");
    }
    const std::size_t bytes = static_cast<std::size_t>(it->expert_stride_bytes);
    if (destination_capacity < bytes) {
        return make_error(ReadStatus::BufferTooSmall, 0, "destination buffer is smaller than one expert slice");
    }

    std::uint64_t expert_offset = 0;
    std::uint64_t read_offset = 0;
    std::uint64_t tensor_end = 0;
    if (!checked_mul(expert_id, it->expert_stride_bytes, expert_offset) ||
            !checked_add(it->absolute_offset, expert_offset, read_offset) ||
            !checked_add(it->absolute_offset, it->tensor_bytes, tensor_end) ||
            expert_offset > it->tensor_bytes || it->expert_stride_bytes > it->tensor_bytes - expert_offset ||
            !fits_off_t(read_offset) || !fits_off_t(tensor_end)) {
        return make_error(ReadStatus::InvalidDescriptor, 0, "expert read offset overflowed its tensor range");
    }

    const Shard & shard = shards_[it->shard_index];
    ReadError file_status = validate_current_file(shard.handle.value, shard.path.c_str(), shard.expected_size, shard.opened_identity);
    if (!file_status.ok()) {
        return file_status;
    }

    std::size_t complete = 0;
    while (complete < bytes) {
        if (cancelled && cancelled->load(std::memory_order_relaxed)) {
            return make_error(ReadStatus::Cancelled, 0, "expert read was cancelled");
        }

        const std::size_t remaining = bytes - complete;
        const std::size_t request = std::min<std::size_t>(remaining,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        std::uint64_t current_offset = 0;
        if (!checked_add(read_offset, complete, current_offset) || !fits_off_t(current_offset)) {
            return make_error(ReadStatus::InvalidDescriptor, 0, "expert read offset does not fit off_t");
        }

        ssize_t count;
#ifdef PS5LM_MOE_EXPERT_SOURCE_TESTING
        if (pread_for_test_) {
            count = pread_for_test_(shard.handle.value,
                    static_cast<std::uint8_t *>(destination) + complete,
                    request, static_cast<off_t>(current_offset), pread_user_data_);
        } else
#endif
        {
            count = ::pread(shard.handle.value,
                    static_cast<std::uint8_t *>(destination) + complete,
                    request, static_cast<off_t>(current_offset));
        }

        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return make_error(ReadStatus::IoError, errno, "pread failed for expert slice");
        }
        if (count == 0) {
            return make_error(ReadStatus::ShortRead, 0, "pread reached end of shard before the expert slice was complete");
        }
        if (static_cast<std::size_t>(count) > request) {
            return make_error(ReadStatus::IoError, 0, "pread returned more bytes than requested");
        }
        complete += static_cast<std::size_t>(count);
    }

    return {};
}

const char * read_status_name(ReadStatus status) noexcept {
    switch (status) {
        case ReadStatus::Ok: return "ok";
        case ReadStatus::InvalidArgument: return "invalid_argument";
        case ReadStatus::InvalidDescriptor: return "invalid_descriptor";
        case ReadStatus::TooManyShards: return "too_many_shards";
        case ReadStatus::TooManyTensors: return "too_many_tensors";
        case ReadStatus::FileOpenFailed: return "file_open_failed";
        case ReadStatus::FileIdentityMismatch: return "file_identity_mismatch";
        case ReadStatus::FileChanged: return "file_changed";
        case ReadStatus::TensorNotFound: return "tensor_not_found";
        case ReadStatus::ExpertOutOfRange: return "expert_out_of_range";
        case ReadStatus::BufferTooSmall: return "buffer_too_small";
        case ReadStatus::BufferLimitExceeded: return "buffer_limit_exceeded";
        case ReadStatus::Cancelled: return "cancelled";
        case ReadStatus::ShortRead: return "short_read";
        case ReadStatus::IoError: return "io_error";
        case ReadStatus::OutOfMemory: return "out_of_memory";
    }
    return "unknown";
}

}  // namespace ps5lm::moe
