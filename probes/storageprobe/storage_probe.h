/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef PS5LM_STORAGE_PROBE_H
#define PS5LM_STORAGE_PROBE_H

#include <stdint.h>
#include <stdio.h>

#define STORAGE_PROBE_READ_SIZE_COUNT 3
#define STORAGE_PROBE_PATTERN_COUNT 2
#define STORAGE_PROBE_MAX_SAMPLES 32
#define STORAGE_PROBE_MAX_BUFFER_BYTES (1024u * 1024u)
#define STORAGE_PROBE_DEFAULT_BYTES (64ull * 1024ull * 1024ull)
#define STORAGE_PROBE_MAX_BYTES (80ull * 1024ull * 1024ull)
#define STORAGE_PROBE_DEFAULT_TIME_MS 30000u
#define STORAGE_PROBE_MAX_TIME_MS 60000u
#define STORAGE_PROBE_MAX_PATH_BYTES 4096
#define STORAGE_PROBE_CONTEXT_BYTES 24

typedef enum storage_probe_status {
    STORAGE_PROBE_ERROR = 0,
    STORAGE_PROBE_COMPLETE,
    STORAGE_PROBE_CANCELLED,
    STORAGE_PROBE_TIME_LIMIT,
} storage_probe_status_t;

typedef enum storage_probe_error {
    STORAGE_PROBE_ERROR_NONE = 0,
    STORAGE_PROBE_ERROR_INVALID_OPTIONS,
    STORAGE_PROBE_ERROR_PATH_TOO_LONG,
    STORAGE_PROBE_ERROR_OPEN,
    STORAGE_PROBE_ERROR_STAT,
    STORAGE_PROBE_ERROR_NOT_REGULAR,
    STORAGE_PROBE_ERROR_BYTE_LIMIT,
    STORAGE_PROBE_ERROR_ALLOCATE,
    STORAGE_PROBE_ERROR_CLOCK,
    STORAGE_PROBE_ERROR_READ,
    STORAGE_PROBE_ERROR_SHORT_READ,
} storage_probe_error_t;

typedef enum storage_probe_group_status {
    STORAGE_PROBE_GROUP_NOT_STARTED = 0,
    STORAGE_PROBE_GROUP_COMPLETE,
    STORAGE_PROBE_GROUP_SKIPPED_FILE_TOO_SMALL,
    STORAGE_PROBE_GROUP_CANCELLED,
    STORAGE_PROBE_GROUP_TIME_LIMIT,
    STORAGE_PROBE_GROUP_READ_ERROR,
} storage_probe_group_status_t;

typedef struct storage_probe_options {
    uint64_t seed;
    uint64_t max_logical_bytes;
    uint32_t time_limit_ms;
    uint32_t samples_per_group;
} storage_probe_options_t;

typedef struct storage_probe_sample {
    uint64_t offset_bytes;
    uint32_t bytes_returned;
    int error_number;
    double latency_ms;
} storage_probe_sample_t;

typedef struct storage_probe_group {
    uint32_t read_size_bytes;
    uint32_t planned_samples;
    uint32_t attempted_samples;
    uint32_t completed_samples;
    uint64_t logical_read_bytes;
    double elapsed_ms;
    storage_probe_group_status_t status;
    int error_number;
    storage_probe_sample_t samples[STORAGE_PROBE_MAX_SAMPLES];
} storage_probe_group_t;

typedef struct storage_probe_report {
    char file_path[STORAGE_PROBE_MAX_PATH_BYTES];
    char execution_context[STORAGE_PROBE_CONTEXT_BYTES];
    uint64_t seed;
    uint64_t file_size_bytes;
    uint64_t logical_read_bytes;
    uint64_t max_logical_bytes;
    uint32_t time_limit_ms;
    uint32_t samples_per_group;
    uint32_t max_concurrent_reads;
    uint32_t read_buffer_bytes;
    double elapsed_ms;
    storage_probe_status_t status;
    storage_probe_error_t error;
    int error_number;
    storage_probe_group_t groups[STORAGE_PROBE_READ_SIZE_COUNT][STORAGE_PROBE_PATTERN_COUNT];
} storage_probe_report_t;

typedef int (*storage_probe_cancel_fn)(void *context);

#ifdef __cplusplus
extern "C" {
#endif

void storage_probe_options_default(storage_probe_options_t *options);

/*
 * Opens only the supplied path with O_RDONLY|O_NONBLOCK and rejects
 * non-regular files (including FIFOs, without waiting for a writer).
 * A cancellation callback is checked between pread calls, including partial
 * reads. The caller owns the callback's context and may pass NULL for both.
 *
 * Returns 0 for complete, 1 for cancelled, 2 for the time limit, and -1 for
 * an error. In every case, report contains the machine-readable outcome.
 */
int storage_probe_run(const char *file_path, const storage_probe_options_t *options,
                      storage_probe_cancel_fn is_cancelled, void *cancel_context,
                      storage_probe_report_t *report);

/* Writes one JSON object. The caller adds a newline and chooses the sink. */
void storage_probe_write_json(FILE *out, const storage_probe_report_t *report);

const char *storage_probe_status_name(storage_probe_status_t status);
const char *storage_probe_error_name(storage_probe_error_t error);
const char *storage_probe_group_status_name(storage_probe_group_status_t status);

#ifdef __cplusplus
}
#endif

#endif
