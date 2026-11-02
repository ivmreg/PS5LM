#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef PS5LM_MOE_EXPERT_SOURCE_TESTING
#include <sys/types.h>
#endif

namespace ps5lm::moe {

enum class ReadStatus : std::uint8_t {
    Ok,
    InvalidArgument,
    InvalidDescriptor,
    TooManyShards,
    TooManyTensors,
    FileOpenFailed,
    FileIdentityMismatch,
    FileChanged,
    TensorNotFound,
    ExpertOutOfRange,
    BufferTooSmall,
    BufferLimitExceeded,
    Cancelled,
    ShortRead,
    IoError,
    OutOfMemory,
};

struct ReadError {
    ReadStatus status = ReadStatus::Ok;
    int system_error = 0;
    char message[160]{};

    bool ok() const noexcept { return status == ReadStatus::Ok; }
};

// POSIX device/inode identity. Set available only when the producer could read
// both values from the source file during metadata validation.
struct FileIdentity {
    bool available = false;
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
};

struct ShardPath {
    std::string path;
};

// Descriptors must come from validated GGUF metadata. Every descriptor for the
// same shard must carry the same expected size and identity.
struct ExpertTensorDescriptor {
    std::uint64_t tensor_id = 0;
    std::uint32_t shard_index = 0;
    std::uint64_t absolute_offset = 0;
    std::uint64_t tensor_bytes = 0;
    std::uint64_t expert_count = 0;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t expected_file_size = 0;
    FileIdentity expected_identity{};
};

struct OpenOptions {
    // Hard upper bound on one expert slice passed to read_expert. This is a
    // caller-owned staging limit; ExpertSource does not allocate that buffer.
    std::size_t max_expert_bytes = 0;

#ifdef PS5LM_MOE_EXPERT_SOURCE_TESTING
    // Fault-injection seam for deterministic short-read and EINTR tests.
    using PreadFunction = ssize_t (*)(int, void *, std::size_t, off_t, void *);
    PreadFunction pread_for_test = nullptr;
    void * pread_user_data = nullptr;
#endif
};

struct OpenResult;

class ExpertSource final {
public:
    static constexpr std::size_t kMaxShards = 64;
    static constexpr std::size_t kMaxTensorDescriptors = 4096;
    static constexpr std::size_t kMaxPathBytes = 4096;

    ~ExpertSource() noexcept;
    ExpertSource(const ExpertSource &) = delete;
    ExpertSource & operator=(const ExpertSource &) = delete;

    // Opens each shard read-only and validates its current identity and exact
    // length before retaining the handle. No model/expert payload is allocated.
    static OpenResult open(
            const std::vector<ShardPath> & shards,
            const std::vector<ExpertTensorDescriptor> & tensors,
            OpenOptions options) noexcept;

    // Reads exactly one quantized expert slice into caller-owned storage. On
    // any failure, the destination may contain partial bytes and must be
    // treated as invalid. Cancellation is checked between pread calls.
    ReadError read_expert(
            std::uint64_t tensor_id,
            std::uint64_t expert_id,
            void * destination,
            std::size_t destination_capacity,
            const std::atomic_bool * cancelled = nullptr) const noexcept;

    std::size_t max_expert_bytes() const noexcept { return max_expert_bytes_; }
    std::size_t tensor_count() const noexcept { return tensors_.size(); }

private:
    struct Shard;

    ExpertSource() noexcept;

    std::vector<Shard> shards_;
    std::vector<ExpertTensorDescriptor> tensors_;
    std::size_t max_expert_bytes_ = 0;
#ifdef PS5LM_MOE_EXPERT_SOURCE_TESTING
    OpenOptions::PreadFunction pread_for_test_ = nullptr;
    void * pread_user_data_ = nullptr;
#endif
};

struct OpenResult {
    std::unique_ptr<ExpertSource> source;
    ReadError error;

    bool ok() const noexcept { return source != nullptr && error.ok(); }
};

const char * read_status_name(ReadStatus status) noexcept;

}  // namespace ps5lm::moe
