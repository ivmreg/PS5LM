// Host test:
//   c++ -std=c++17 -O1 -DPS5LM_MOE_EXPERT_SOURCE_TESTING ps5/moe/expert_source_test.cpp ps5/moe/expert_source.cpp -o /tmp/ps5lm_expert_source_test
//   /tmp/ps5lm_expert_source_test

#include "expert_source.hpp"

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
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace ps5lm::moe;

namespace {

struct TempDir {
    std::string path;
    std::vector<std::string> files;

    TempDir() {
        char pattern[] = "/tmp/ps5lm_expert_source_XXXXXX";
        char * created = ::mkdtemp(pattern);
        assert(created != nullptr);
        path = created;
    }

    ~TempDir() {
        if (!path.empty()) {
            for (const std::string & file_path : files) {
                ::unlink(file_path.c_str());
            }
            ::rmdir(path.c_str());
        }
    }

    std::string file(const char * name) {
        std::string result = path + "/" + name;
        files.push_back(result);
        return result;
    }
};

void write_file(const std::string & path, const std::vector<std::uint8_t> & bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    assert(fd >= 0);
    std::size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
        assert(n > 0);
        written += static_cast<std::size_t>(n);
    }
    assert(::close(fd) == 0);
}

void write_replacement(const std::string & path, std::uint8_t seed, std::size_t size) {
    std::vector<std::uint8_t> bytes(size);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<std::uint8_t>(seed + i);
    }
    write_file(path, bytes);
}

FileIdentity file_identity(const std::string & path) {
    struct stat st{};
    assert(::stat(path.c_str(), &st) == 0);
    FileIdentity identity;
    identity.device = static_cast<std::uint64_t>(st.st_dev);
    identity.inode = static_cast<std::uint64_t>(st.st_ino);
    identity.available = st.st_ino != 0;
    return identity;
}

struct ReadScript {
    std::size_t max_chunk = std::numeric_limits<std::size_t>::max();
    int calls = 0;
    bool inject_eintr = false;
    bool return_zero_after_first_chunk = false;
    std::atomic_bool * cancel_after_first_chunk = nullptr;
    std::uint64_t first_offset = 0;
    std::size_t bytes_requested = 0;
};

struct MetadataScript {
    int stat_calls = 0;
    int fstat_calls = 0;
    bool report_identity_unavailable = false;

    void reset() {
        stat_calls = 0;
        fstat_calls = 0;
    }
};

int counted_stat(const char * path, struct stat * output, void * user_data) {
    auto * script = static_cast<MetadataScript *>(user_data);
    ++script->stat_calls;
    return ::stat(path, output);
}

int counted_fstat(int fd, struct stat * output, void * user_data) {
    auto * script = static_cast<MetadataScript *>(user_data);
    ++script->fstat_calls;
    const int result = ::fstat(fd, output);
    if (result == 0 && script->report_identity_unavailable) {
        output->st_dev = 0;
        output->st_ino = 0;
    }
    return result;
}

ssize_t scripted_pread(int fd, void * output, std::size_t count, off_t offset, void * user_data) {
    auto * script = static_cast<ReadScript *>(user_data);
    const int call = script->calls++;
    if (script->inject_eintr && call == 0) {
        errno = EINTR;
        return -1;
    }
    if (script->return_zero_after_first_chunk && call > 0) {
        return 0;
    }

    const std::size_t request = std::min(count, script->max_chunk);
    if (call == 0 || (script->inject_eintr && call == 1)) {
        script->first_offset = static_cast<std::uint64_t>(offset);
    }
    script->bytes_requested += request;
    const ssize_t read_count = ::pread(fd, output, request, offset);
    if (read_count > 0 && script->cancel_after_first_chunk) {
        script->cancel_after_first_chunk->store(true, std::memory_order_relaxed);
        script->cancel_after_first_chunk = nullptr;
    }
    return read_count;
}

struct Fixture {
    TempDir dir;
    std::string shard_a = dir.file("a.gguf");
    std::string shard_b = dir.file("b.gguf");
    std::vector<std::uint8_t> bytes_a;
    std::vector<std::uint8_t> bytes_b;

    Fixture() {
        bytes_a.resize(32);
        bytes_b.resize(24);
        for (std::size_t i = 0; i < bytes_a.size(); ++i) {
            bytes_a[i] = static_cast<std::uint8_t>(i + 1);
        }
        for (std::size_t i = 0; i < bytes_b.size(); ++i) {
            bytes_b[i] = static_cast<std::uint8_t>(101 + i);
        }
        write_file(shard_a, bytes_a);
        write_file(shard_b, bytes_b);
    }

    std::vector<ShardPath> shards() const {
        return {{shard_a}, {shard_b}};
    }

    std::vector<ExpertTensorDescriptor> descriptors() const {
        return {
            {11, 0, 4, 12, 3, 4, bytes_a.size(), file_identity(shard_a)},
            {22, 0, 16, 8, 2, 4, bytes_a.size(), file_identity(shard_a)},
            {33, 1, 2, 16, 2, 8, bytes_b.size(), file_identity(shard_b)},
        };
    }

    OpenResult open(OpenOptions options = {}) const {
        options.max_expert_bytes = options.max_expert_bytes ? options.max_expert_bytes : 8;
        return ExpertSource::open(shards(), descriptors(), options);
    }
};

void expect_status(const ReadError & error, ReadStatus expected) {
    if (error.status != expected) {
        std::cerr << "expected " << read_status_name(expected) << ", got "
                  << read_status_name(error.status) << ": " << error.message << '\n';
        std::abort();
    }
}

void test_exact_reads_from_multiple_shards() {
    Fixture fixture;
    OpenOptions options;
    options.max_expert_bytes = 8;
    ReadScript script;
    script.max_chunk = 2;
    options.pread_for_test = scripted_pread;
    options.pread_user_data = &script;
    OpenResult opened = fixture.open(options);
    assert(opened.ok());
    assert(opened.source->tensor_count() == 3);
    assert(opened.source->max_expert_bytes() == 8);

    std::array<std::uint8_t, 10> guarded{};
    guarded.front() = 0xA5;
    guarded.back() = 0x5A;
    ReadError error = opened.source->read_expert(11, 2, guarded.data() + 1, 4);
    assert(error.ok());
    assert((std::array<std::uint8_t, 4>{guarded[1], guarded[2], guarded[3], guarded[4]} ==
            std::array<std::uint8_t, 4>{13, 14, 15, 16}));
    assert(guarded.front() == 0xA5 && guarded.back() == 0x5A);
    assert(script.first_offset == 12);
    assert(script.bytes_requested == 4);
    assert(script.calls == 2);

    std::array<std::uint8_t, 8> second{};
    error = opened.source->read_expert(33, 1, second.data(), second.size());
    assert(error.ok());
    assert((second == std::array<std::uint8_t, 8>{111, 112, 113, 114, 115, 116, 117, 118}));
}

void test_eintr_retry_and_real_short_read() {
    Fixture fixture;
    ReadScript script;
    script.max_chunk = 2;
    script.inject_eintr = true;
    OpenOptions options;
    options.max_expert_bytes = 8;
    options.pread_for_test = scripted_pread;
    options.pread_user_data = &script;
    OpenResult opened = fixture.open(options);
    assert(opened.ok());

    std::array<std::uint8_t, 4> bytes{};
    ReadError error = opened.source->read_expert(11, 1, bytes.data(), bytes.size());
    assert(error.ok());
    assert((bytes == std::array<std::uint8_t, 4>{9, 10, 11, 12}));
    assert(script.calls == 3);

    ReadScript short_script;
    short_script.max_chunk = 2;
    short_script.return_zero_after_first_chunk = true;
    options.pread_user_data = &short_script;
    opened = fixture.open(options);
    assert(opened.ok());
    error = opened.source->read_expert(11, 1, bytes.data(), bytes.size());
    expect_status(error, ReadStatus::ShortRead);
}

void test_cancellation() {
    Fixture fixture;
    ReadScript script;
    script.max_chunk = 2;
    std::atomic_bool cancelled{false};
    script.cancel_after_first_chunk = &cancelled;
    OpenOptions options;
    options.max_expert_bytes = 8;
    options.pread_for_test = scripted_pread;
    options.pread_user_data = &script;
    OpenResult opened = fixture.open(options);
    assert(opened.ok());

    std::array<std::uint8_t, 8> bytes{};
    ReadError error = opened.source->read_expert(33, 0, bytes.data(), bytes.size(), &cancelled);
    expect_status(error, ReadStatus::Cancelled);

    cancelled.store(true, std::memory_order_relaxed);
    error = opened.source->read_expert(33, 0, bytes.data(), bytes.size(), &cancelled);
    expect_status(error, ReadStatus::Cancelled);
}

void test_descriptor_and_buffer_validation() {
    Fixture fixture;
    OpenOptions options;
    options.max_expert_bytes = 3;
    OpenResult opened = fixture.open(options);
    expect_status(opened.error, ReadStatus::BufferLimitExceeded);

    options.max_expert_bytes = 8;
    opened = fixture.open(options);
    assert(opened.ok());
    std::array<std::uint8_t, 7> bytes{};
    expect_status(opened.source->read_expert(33, 0, bytes.data(), 7), ReadStatus::BufferTooSmall);
    expect_status(opened.source->read_expert(999, 0, bytes.data(), bytes.size()), ReadStatus::TensorNotFound);
    expect_status(opened.source->read_expert(33, 2, bytes.data(), bytes.size()), ReadStatus::ExpertOutOfRange);
    expect_status(opened.source->read_expert(33, 0, nullptr, bytes.size()), ReadStatus::InvalidArgument);

    auto descriptors = fixture.descriptors();
    descriptors[0].expert_count = std::numeric_limits<std::uint64_t>::max();
    descriptors[0].expert_stride_bytes = 2;
    descriptors[0].tensor_bytes = 8;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);

    descriptors = fixture.descriptors();
    descriptors[0].absolute_offset = std::numeric_limits<std::uint64_t>::max() - 3;
    descriptors[0].tensor_bytes = 12;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);

    descriptors = fixture.descriptors();
    descriptors[0].absolute_offset = 31;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);

    descriptors = fixture.descriptors();
    descriptors[1].expected_file_size += 1;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);

    descriptors = fixture.descriptors();
    descriptors[0].tensor_id = descriptors[1].tensor_id;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);
}

void test_identity_mismatch_and_open_length_mismatch() {
    Fixture fixture;
    auto descriptors = fixture.descriptors();
    descriptors[0].expected_identity.inode += 1;
    descriptors[1].expected_identity = descriptors[0].expected_identity;
    OpenOptions options;
    options.max_expert_bytes = 8;
    OpenResult opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::FileIdentityMismatch);

    descriptors = fixture.descriptors();
    descriptors[0].expected_file_size += 1;
    descriptors[1].expected_file_size += 1;
    opened = ExpertSource::open(fixture.shards(), descriptors, options);
    expect_status(opened.error, ReadStatus::FileChanged);
}

void test_read_validation_uses_one_path_check_when_identity_is_available() {
    Fixture fixture;
    MetadataScript metadata;
    OpenOptions options;
    options.max_expert_bytes = 8;
    options.stat_for_test = counted_stat;
    options.fstat_for_test = counted_fstat;
    options.metadata_user_data = &metadata;
    OpenResult opened = fixture.open(options);
    assert(opened.ok());

    metadata.reset();
    std::array<std::uint8_t, 4> bytes{};
    const ReadError error = opened.source->read_expert(11, 1, bytes.data(), bytes.size());
    assert(error.ok());
    assert((bytes == std::array<std::uint8_t, 4>{9, 10, 11, 12}));
    assert(metadata.stat_calls == 1);
    assert(metadata.fstat_calls == 0);
}

void test_read_validation_keeps_fd_size_check_without_identity() {
    Fixture fixture;
    MetadataScript metadata;
    metadata.report_identity_unavailable = true;
    OpenOptions options;
    options.max_expert_bytes = 8;
    options.stat_for_test = counted_stat;
    options.fstat_for_test = counted_fstat;
    options.metadata_user_data = &metadata;
    std::vector<ExpertTensorDescriptor> descriptors = fixture.descriptors();
    for (ExpertTensorDescriptor & descriptor : descriptors) {
        descriptor.expected_identity = {};
    }
    OpenResult opened = ExpertSource::open(fixture.shards(), descriptors, options);
    assert(opened.ok());

    metadata.reset();
    std::array<std::uint8_t, 4> bytes{};
    const ReadError error = opened.source->read_expert(11, 1, bytes.data(), bytes.size());
    assert(error.ok());
    assert((bytes == std::array<std::uint8_t, 4>{9, 10, 11, 12}));
    assert(metadata.stat_calls == 0);
    assert(metadata.fstat_calls == 1);
}

void test_truncation_and_path_replacement_after_open() {
    {
        Fixture fixture;
        OpenOptions options;
        options.max_expert_bytes = 8;
        OpenResult opened = fixture.open(options);
        assert(opened.ok());
        assert(::truncate(fixture.shard_a.c_str(), 20) == 0);
        std::array<std::uint8_t, 4> bytes{};
        expect_status(opened.source->read_expert(11, 0, bytes.data(), bytes.size()), ReadStatus::FileChanged);
    }

    {
        Fixture fixture;
        OpenOptions options;
        options.max_expert_bytes = 8;
        OpenResult opened = fixture.open(options);
        assert(opened.ok());
        const std::string replacement = fixture.dir.file("replacement.gguf");
        write_replacement(replacement, 200, fixture.bytes_a.size());
        assert(::rename(replacement.c_str(), fixture.shard_a.c_str()) == 0);
        std::array<std::uint8_t, 4> bytes{};
        expect_status(opened.source->read_expert(11, 0, bytes.data(), bytes.size()), ReadStatus::FileChanged);
    }
}

void test_no_source_mmap_or_unbounded_read_contract() {
    Fixture fixture;
    ReadScript script;
    script.max_chunk = 2;
    OpenOptions options;
    options.max_expert_bytes = 8;
    options.pread_for_test = scripted_pread;
    options.pread_user_data = &script;
    OpenResult opened = fixture.open(options);
    assert(opened.ok());

    std::array<std::uint8_t, 8> bytes{};
    ReadError error = opened.source->read_expert(11, 0, bytes.data(), bytes.size());
    assert(error.ok());
    assert(script.bytes_requested == 4);
    assert(script.bytes_requested < 12); // tensor_bytes: only one expert was read
}

void test_rejects_non_regular_and_embedded_nul_paths() {
    Fixture fixture;
    OpenOptions options;
    options.max_expert_bytes = 8;
    const std::string fifo = fixture.dir.file("fifo");
    assert(::mkfifo(fifo.c_str(), 0600) == 0);
    auto paths = fixture.shards();
    paths[0].path = fifo;
    auto descriptors = fixture.descriptors();
    descriptors[0].expected_identity = {};
    descriptors[1].expected_identity = {};
    OpenResult opened = ExpertSource::open(paths, descriptors, options);
    expect_status(opened.error, ReadStatus::InvalidDescriptor);

    paths = fixture.shards();
    paths[0].path += std::string("\0suffix", 7);
    opened = ExpertSource::open(paths, fixture.descriptors(), options);
    expect_status(opened.error, ReadStatus::InvalidArgument);
}

}  // namespace

int main() {
    test_exact_reads_from_multiple_shards();
    test_eintr_retry_and_real_short_read();
    test_cancellation();
    test_descriptor_and_buffer_validation();
    test_identity_mismatch_and_open_length_mismatch();
    test_read_validation_uses_one_path_check_when_identity_is_available();
    test_read_validation_keeps_fd_size_check_without_identity();
    test_truncation_and_path_replacement_after_open();
    test_no_source_mmap_or_unbounded_read_contract();
    test_rejects_non_regular_and_embedded_nul_paths();
    std::cout << "expert_source_test: PASS\n";
}
