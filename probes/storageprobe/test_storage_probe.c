/* Deterministic host-side tests using a small synthetic regular file. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L

#include "storage_probe.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MiB (1024u * 1024u)
#define FIXTURE_SIZE (8u * MiB)

static void fill_fixture(int fd, uint64_t size) {
    unsigned char bytes[16 * 1024];
    uint64_t offset = 0;
    while (offset < size) {
        size_t n = sizeof(bytes);
        if (size - offset < n)
            n = (size_t)(size - offset);
        for (size_t i = 0; i < n; ++i)
            bytes[i] = (unsigned char)(((offset + i) * 131u + 17u) & 0xffu);
        assert(write(fd, bytes, n) == (ssize_t)n);
        offset += n;
    }
    assert(fsync(fd) == 0);
}

static void verify_fixture_unchanged(const char *path) {
    int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    unsigned char bytes[16 * 1024];
    uint64_t offset = 0;
    while (offset < FIXTURE_SIZE) {
        size_t n = sizeof(bytes);
        if (FIXTURE_SIZE - offset < n)
            n = FIXTURE_SIZE - (size_t)offset;
        assert(pread(fd, bytes, n, (off_t)offset) == (ssize_t)n);
        for (size_t i = 0; i < n; ++i)
            assert(bytes[i] == (unsigned char)(((offset + i) * 131u + 17u) & 0xffu));
        offset += n;
    }
    close(fd);
}

static int cancel_after(void *context) {
    unsigned *calls = context;
    const unsigned current = (*calls)++;
    return current >= 3;
}

static void check_json(const storage_probe_report_t *report) {
    FILE *file = tmpfile();
    assert(file != NULL);
    storage_probe_write_json(file, report);
    assert(fflush(file) == 0);
    assert(fseek(file, 0, SEEK_END) == 0);
    const long size = ftell(file);
    assert(size > 0 && size < 1024 * 1024);
    assert(fseek(file, 0, SEEK_SET) == 0);
    char *json = malloc((size_t)size + 1);
    assert(json != NULL);
    assert(fread(json, 1, (size_t)size, file) == (size_t)size);
    json[size] = '\0';
    assert(strstr(json, "\"cache_state\":\"unknown\"") != NULL);
    assert(strstr(json, "\"physical_read_bytes\":null") != NULL);
    assert(strstr(json, "\"execution_context\":\"unspecified\"") != NULL);
    assert(strstr(json, "\"read_accounting\":\"bytes_returned_by_pread\"") != NULL);
    assert(strstr(json, "\"max_concurrent_reads\":1") != NULL);
    assert(strstr(json, "\"access_pattern\":\"sequential\"") != NULL);
    assert(strstr(json, "\"access_pattern\":\"random\"") != NULL);
    free(json);
    fclose(file);
}

int main(void) {
    char fixture_path[] = "/tmp/ps5lm-storageprobe-XXXXXX";
    const int fixture_fd = mkstemp(fixture_path);
    assert(fixture_fd >= 0);
    fill_fixture(fixture_fd, FIXTURE_SIZE);
    close(fixture_fd);

    storage_probe_options_t options;
    storage_probe_options_default(&options);
    options.samples_per_group = 4;
    options.max_logical_bytes = 16ull * MiB;
    options.seed = 0x123456789abcdef0ull;

    storage_probe_report_t first;
    assert(storage_probe_run(fixture_path, &options, NULL, NULL, &first) == 0);
    assert(first.status == STORAGE_PROBE_COMPLETE);
    assert(first.error == STORAGE_PROBE_ERROR_NONE);
    assert(first.file_size_bytes == FIXTURE_SIZE);
    assert(first.max_concurrent_reads == 1);
    assert(first.read_buffer_bytes == STORAGE_PROBE_MAX_BUFFER_BYTES);
    assert(first.logical_read_bytes == 8ull * (4096ull + 65536ull + 1048576ull));
    for (unsigned size_index = 0; size_index < STORAGE_PROBE_READ_SIZE_COUNT; ++size_index) {
        const uint32_t read_size = first.groups[size_index][0].read_size_bytes;
        const storage_probe_group_t *sequential = &first.groups[size_index][0];
        const storage_probe_group_t *random = &first.groups[size_index][1];
        assert(sequential->status == STORAGE_PROBE_GROUP_COMPLETE);
        assert(random->status == STORAGE_PROBE_GROUP_COMPLETE);
        assert(sequential->planned_samples == 4 && sequential->completed_samples == 4);
        assert(random->planned_samples == 4 && random->completed_samples == 4);
        for (unsigned sample = 1; sample < sequential->attempted_samples; ++sample)
            assert(sequential->samples[sample].offset_bytes == sequential->samples[sample - 1].offset_bytes + read_size);
        for (unsigned sample = 0; sample < random->attempted_samples; ++sample) {
            assert(random->samples[sample].offset_bytes <= FIXTURE_SIZE - read_size);
            assert(random->samples[sample].bytes_returned == read_size);
        }
    }

    storage_probe_report_t replay;
    assert(storage_probe_run(fixture_path, &options, NULL, NULL, &replay) == 0);
    for (unsigned i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i) {
        for (unsigned p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p) {
            assert(first.groups[i][p].attempted_samples == replay.groups[i][p].attempted_samples);
            for (unsigned sample = 0; sample < first.groups[i][p].attempted_samples; ++sample)
                assert(first.groups[i][p].samples[sample].offset_bytes == replay.groups[i][p].samples[sample].offset_bytes);
        }
    }
    check_json(&first);
    verify_fixture_unchanged(fixture_path);

    unsigned cancel_calls = 0;
    storage_probe_report_t cancelled;
    assert(storage_probe_run(fixture_path, &options, cancel_after, &cancel_calls, &cancelled) == 1);
    assert(cancelled.status == STORAGE_PROBE_CANCELLED);
    assert(cancelled.groups[0][0].attempted_samples == 3);
    assert(cancelled.logical_read_bytes == 3ull * 4096ull);

    storage_probe_options_t too_small_budget = options;
    too_small_budget.max_logical_bytes = 4096;
    storage_probe_report_t budget_report;
    assert(storage_probe_run(fixture_path, &too_small_budget, NULL, NULL, &budget_report) == -1);
    assert(budget_report.error == STORAGE_PROBE_ERROR_BYTE_LIMIT);
    assert(budget_report.logical_read_bytes == 0);

    storage_probe_report_t missing;
    assert(storage_probe_run("/tmp/ps5lm-storageprobe-file-does-not-exist", &options, NULL, NULL, &missing) == -1);
    assert(missing.error == STORAGE_PROBE_ERROR_OPEN);
    assert(missing.logical_read_bytes == 0);

    storage_probe_report_t directory;
    assert(storage_probe_run("/tmp", &options, NULL, NULL, &directory) == -1);
    assert(directory.error == STORAGE_PROBE_ERROR_NOT_REGULAR);

    char fifo_path[sizeof(fixture_path) + 8];
    snprintf(fifo_path, sizeof(fifo_path), "%s.fifo", fixture_path);
    assert(mkfifo(fifo_path, 0600) == 0);
    alarm(5);  // A blocking FIFO open would fail this test instead of hanging CI.
    storage_probe_report_t fifo;
    const int fifo_rc = storage_probe_run(fifo_path, &options, NULL, NULL, &fifo);
    alarm(0);
    assert(fifo_rc == -1);
    assert(fifo.error == STORAGE_PROBE_ERROR_NOT_REGULAR);
    unlink(fifo_path);

    char small_path[] = "/tmp/ps5lm-storageprobe-small-XXXXXX";
    const int small_fd = mkstemp(small_path);
    assert(small_fd >= 0);
    assert(write(small_fd, "fixture", 7) == 7);
    close(small_fd);
    storage_probe_report_t small;
    assert(storage_probe_run(small_path, &options, NULL, NULL, &small) == 0);
    assert(small.logical_read_bytes == 0);
    for (unsigned i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i)
        for (unsigned p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p)
            assert(small.groups[i][p].status == STORAGE_PROBE_GROUP_SKIPPED_FILE_TOO_SMALL);
    unlink(small_path);

    unlink(fixture_path);
    puts("storage_probe_test: ok (bounded reads, seeded offsets, cancellation, errors, FIFO, JSON, read-only fixture)");
    return 0;
}
