/* SPDX-License-Identifier: GPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L

#include "storage_probe.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const uint32_t k_read_sizes[STORAGE_PROBE_READ_SIZE_COUNT] = {
    4u * 1024u,
    64u * 1024u,
    STORAGE_PROBE_MAX_BUFFER_BYTES,
};

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return -1.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static uint32_t planned_samples(uint64_t file_size, uint32_t read_size, uint32_t requested,
                               int sequential) {
    const uint64_t slots = file_size / read_size;
    if (slots == 0)
        return 0;
    if (sequential && slots < requested)
        return (uint32_t)slots;
    return requested;
}

static void set_error(storage_probe_report_t *report, storage_probe_error_t error, int error_number) {
    report->status = STORAGE_PROBE_ERROR;
    report->error = error;
    report->error_number = error_number;
}

static int validate_options(const storage_probe_options_t *options) {
    return options != NULL && options->samples_per_group > 0 &&
           options->samples_per_group <= STORAGE_PROBE_MAX_SAMPLES &&
           options->max_logical_bytes > 0 && options->max_logical_bytes <= STORAGE_PROBE_MAX_BYTES &&
           options->time_limit_ms > 0 && options->time_limit_ms <= STORAGE_PROBE_MAX_TIME_MS;
}

void storage_probe_options_default(storage_probe_options_t *options) {
    if (!options)
        return;
    options->seed = 1;
    options->max_logical_bytes = STORAGE_PROBE_DEFAULT_BYTES;
    options->time_limit_ms = STORAGE_PROBE_DEFAULT_TIME_MS;
    options->samples_per_group = 16;
}

typedef enum read_result {
    READ_RESULT_OK,
    READ_RESULT_CANCELLED,
    READ_RESULT_TIME_LIMIT,
    READ_RESULT_ERROR,
    READ_RESULT_SHORT,
} read_result_t;

static read_result_t read_exact(int fd, unsigned char *buffer, uint32_t read_size, uint64_t offset,
                                double start_ms, uint32_t time_limit_ms,
                                storage_probe_cancel_fn is_cancelled, void *cancel_context,
                                uint32_t *bytes_returned, int *error_number, int *did_io) {
    uint32_t done = 0;
    *bytes_returned = 0;
    *error_number = 0;
    *did_io = 0;

    while (done < read_size) {
        const double now = monotonic_ms();
        if (now < 0.0) {
            *error_number = errno;
            return READ_RESULT_ERROR;
        }
        if (now - start_ms >= (double)time_limit_ms)
            return READ_RESULT_TIME_LIMIT;
        if (is_cancelled && is_cancelled(cancel_context))
            return READ_RESULT_CANCELLED;

        *did_io = 1;
        const ssize_t n = pread(fd, buffer + done, (size_t)(read_size - done), (off_t)(offset + done));
        if (n > 0) {
            done += (uint32_t)n;
            *bytes_returned = done;
            continue;
        }
        if (n == 0)
            return READ_RESULT_SHORT;
        if (errno == EINTR) {
            if (is_cancelled && is_cancelled(cancel_context)) {
                *error_number = EINTR;
                return READ_RESULT_CANCELLED;
            }
            continue;
        }
        *error_number = errno;
        return READ_RESULT_ERROR;
    }
    return READ_RESULT_OK;
}

static int run_group(int fd, unsigned char *buffer, uint64_t file_size, double run_start_ms,
                     const storage_probe_options_t *options, storage_probe_cancel_fn is_cancelled,
                     void *cancel_context, storage_probe_report_t *report, uint32_t size_index,
                     uint32_t pattern_index, uint64_t *random_state) {
    storage_probe_group_t *group = &report->groups[size_index][pattern_index];
    const uint32_t read_size = group->read_size_bytes;
    const int sequential = pattern_index == 0;
    const uint64_t slots = file_size / read_size;
    const double group_start_ms = monotonic_ms();
    if (group_start_ms < 0.0) {
        set_error(report, STORAGE_PROBE_ERROR_CLOCK, errno);
        group->status = STORAGE_PROBE_GROUP_READ_ERROR;
        return -1;
    }

    uint64_t sequential_start_slot = 0;
    if (sequential) {
        const uint64_t latest_start = slots - group->planned_samples;
        sequential_start_slot = next_random(random_state) % (latest_start + 1);
    }

    for (uint32_t sample_index = 0; sample_index < group->planned_samples; ++sample_index) {
        storage_probe_sample_t sample;
        memset(&sample, 0, sizeof(sample));
        const uint64_t slot = sequential ? sequential_start_slot + sample_index : next_random(random_state) % slots;
        sample.offset_bytes = slot * read_size;
        const double sample_start_ms = monotonic_ms();
        if (sample_start_ms < 0.0) {
            group->status = STORAGE_PROBE_GROUP_READ_ERROR;
            set_error(report, STORAGE_PROBE_ERROR_CLOCK, errno);
            goto done;
        }

        int did_io = 0;
        const read_result_t result = read_exact(fd, buffer, read_size, sample.offset_bytes,
                                                run_start_ms, options->time_limit_ms,
                                                is_cancelled, cancel_context,
                                                &sample.bytes_returned, &sample.error_number, &did_io);
        const double sample_end_ms = monotonic_ms();
        if (sample_end_ms < 0.0) {
            group->status = STORAGE_PROBE_GROUP_READ_ERROR;
            set_error(report, STORAGE_PROBE_ERROR_CLOCK, errno);
            goto done;
        }
        sample.latency_ms = sample_end_ms - sample_start_ms;
        if (did_io) {
            group->samples[group->attempted_samples++] = sample;
            group->logical_read_bytes += sample.bytes_returned;
            report->logical_read_bytes += sample.bytes_returned;
        }

        if (result == READ_RESULT_OK) {
            group->completed_samples++;
            continue;
        }
        if (result == READ_RESULT_CANCELLED) {
            group->status = STORAGE_PROBE_GROUP_CANCELLED;
            report->status = STORAGE_PROBE_CANCELLED;
            goto done;
        }
        if (result == READ_RESULT_TIME_LIMIT) {
            group->status = STORAGE_PROBE_GROUP_TIME_LIMIT;
            report->status = STORAGE_PROBE_TIME_LIMIT;
            goto done;
        }
        if (result == READ_RESULT_SHORT) {
            if (did_io) {
                group->samples[group->attempted_samples - 1].error_number = 0;
            }
            group->status = STORAGE_PROBE_GROUP_READ_ERROR;
            set_error(report, STORAGE_PROBE_ERROR_SHORT_READ, 0);
            goto done;
        }
        if (did_io)
            group->error_number = sample.error_number;
        group->status = STORAGE_PROBE_GROUP_READ_ERROR;
        set_error(report, STORAGE_PROBE_ERROR_READ, sample.error_number);
        goto done;
    }
    group->status = STORAGE_PROBE_GROUP_COMPLETE;

done:
    {
        const double group_end_ms = monotonic_ms();
        if (group_end_ms >= group_start_ms)
            group->elapsed_ms = group_end_ms - group_start_ms;
    }
    return report->status == STORAGE_PROBE_ERROR ? -1 :
           report->status == STORAGE_PROBE_CANCELLED ? 1 :
           report->status == STORAGE_PROBE_TIME_LIMIT ? 2 : 0;
}

int storage_probe_run(const char *file_path, const storage_probe_options_t *options,
                      storage_probe_cancel_fn is_cancelled, void *cancel_context,
                      storage_probe_report_t *report) {
    if (!report)
        return -1;
    memset(report, 0, sizeof(*report));
    report->status = STORAGE_PROBE_ERROR;
    report->error = STORAGE_PROBE_ERROR_INVALID_OPTIONS;
    report->max_concurrent_reads = 1;
    report->read_buffer_bytes = STORAGE_PROBE_MAX_BUFFER_BYTES;

    if (!file_path || file_path[0] == '\0' || !validate_options(options))
        return -1;
    const size_t path_len = strlen(file_path);
    if (path_len >= sizeof(report->file_path)) {
        report->error = STORAGE_PROBE_ERROR_PATH_TOO_LONG;
        return -1;
    }
    memcpy(report->file_path, file_path, path_len + 1);
    report->seed = options->seed;
    report->max_logical_bytes = options->max_logical_bytes;
    report->time_limit_ms = options->time_limit_ms;
    report->samples_per_group = options->samples_per_group;
    report->status = STORAGE_PROBE_COMPLETE;
    report->error = STORAGE_PROBE_ERROR_NONE;

    for (uint32_t i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i) {
        for (uint32_t p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p) {
            storage_probe_group_t *group = &report->groups[i][p];
            group->read_size_bytes = k_read_sizes[i];
            group->status = STORAGE_PROBE_GROUP_NOT_STARTED;
        }
    }

    /* Nonblocking open lets fstat reject FIFOs without waiting for a writer. */
    const int fd = open(file_path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        set_error(report, STORAGE_PROBE_ERROR_OPEN, errno);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        const int saved_errno = errno;
        close(fd);
        set_error(report, STORAGE_PROBE_ERROR_STAT, saved_errno);
        return -1;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd);
        set_error(report, STORAGE_PROBE_ERROR_NOT_REGULAR, 0);
        return -1;
    }
    report->file_size_bytes = (uint64_t)st.st_size;

    uint64_t planned_bytes = 0;
    for (uint32_t i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i) {
        for (uint32_t p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p) {
            storage_probe_group_t *group = &report->groups[i][p];
            group->planned_samples = planned_samples(report->file_size_bytes, group->read_size_bytes,
                                                     options->samples_per_group, p == 0);
            if (group->planned_samples == 0) {
                group->status = STORAGE_PROBE_GROUP_SKIPPED_FILE_TOO_SMALL;
                continue;
            }
            planned_bytes += (uint64_t)group->planned_samples * group->read_size_bytes;
        }
    }
    if (planned_bytes > options->max_logical_bytes) {
        close(fd);
        set_error(report, STORAGE_PROBE_ERROR_BYTE_LIMIT, 0);
        return -1;
    }

    unsigned char *buffer = malloc(STORAGE_PROBE_MAX_BUFFER_BYTES);
    if (!buffer) {
        const int saved_errno = errno;
        close(fd);
        set_error(report, STORAGE_PROBE_ERROR_ALLOCATE, saved_errno);
        return -1;
    }

    const double run_start_ms = monotonic_ms();
    if (run_start_ms < 0.0) {
        const int saved_errno = errno;
        free(buffer);
        close(fd);
        set_error(report, STORAGE_PROBE_ERROR_CLOCK, saved_errno);
        return -1;
    }
    uint64_t random_state = options->seed;
    for (uint32_t i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i) {
        for (uint32_t p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p) {
            storage_probe_group_t *group = &report->groups[i][p];
            if (group->status == STORAGE_PROBE_GROUP_SKIPPED_FILE_TOO_SMALL)
                continue;
            if (run_group(fd, buffer, report->file_size_bytes, run_start_ms, options,
                          is_cancelled, cancel_context, report, i, p, &random_state) != 0)
                goto finished;
        }
    }
    report->status = STORAGE_PROBE_COMPLETE;
    report->error = STORAGE_PROBE_ERROR_NONE;
    report->error_number = 0;

finished:
    {
        const double end_ms = monotonic_ms();
        if (end_ms >= run_start_ms) {
            report->elapsed_ms = end_ms - run_start_ms;
            /* A final synchronous read can finish just after the last deadline check. */
            if (report->status == STORAGE_PROBE_COMPLETE &&
                report->elapsed_ms >= (double)report->time_limit_ms)
                report->status = STORAGE_PROBE_TIME_LIMIT;
        }
    }
    free(buffer);
    close(fd);
    switch (report->status) {
        case STORAGE_PROBE_COMPLETE: return 0;
        case STORAGE_PROBE_CANCELLED: return 1;
        case STORAGE_PROBE_TIME_LIMIT: return 2;
        default: return -1;
    }
}

const char *storage_probe_status_name(storage_probe_status_t status) {
    switch (status) {
        case STORAGE_PROBE_COMPLETE: return "complete";
        case STORAGE_PROBE_CANCELLED: return "cancelled";
        case STORAGE_PROBE_TIME_LIMIT: return "time_limit";
        default: return "error";
    }
}

const char *storage_probe_error_name(storage_probe_error_t error) {
    switch (error) {
        case STORAGE_PROBE_ERROR_NONE: return "none";
        case STORAGE_PROBE_ERROR_INVALID_OPTIONS: return "invalid_options";
        case STORAGE_PROBE_ERROR_PATH_TOO_LONG: return "path_too_long";
        case STORAGE_PROBE_ERROR_OPEN: return "open_failed";
        case STORAGE_PROBE_ERROR_STAT: return "stat_failed";
        case STORAGE_PROBE_ERROR_NOT_REGULAR: return "not_regular_file";
        case STORAGE_PROBE_ERROR_BYTE_LIMIT: return "byte_limit_exceeded";
        case STORAGE_PROBE_ERROR_ALLOCATE: return "buffer_allocation_failed";
        case STORAGE_PROBE_ERROR_CLOCK: return "clock_failed";
        case STORAGE_PROBE_ERROR_READ: return "read_failed";
        case STORAGE_PROBE_ERROR_SHORT_READ: return "short_read";
        default: return "unknown_error";
    }
}

const char *storage_probe_group_status_name(storage_probe_group_status_t status) {
    switch (status) {
        case STORAGE_PROBE_GROUP_COMPLETE: return "complete";
        case STORAGE_PROBE_GROUP_SKIPPED_FILE_TOO_SMALL: return "skipped_file_too_small";
        case STORAGE_PROBE_GROUP_CANCELLED: return "cancelled";
        case STORAGE_PROBE_GROUP_TIME_LIMIT: return "time_limit";
        case STORAGE_PROBE_GROUP_READ_ERROR: return "read_error";
        default: return "not_started";
    }
}

static void write_json_string(FILE *out, const char *value) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        switch (*p) {
            case '"': fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\b': fputs("\\b", out); break;
            case '\f': fputs("\\f", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (*p < 0x20 || *p >= 0x7f) {
                    fputs("\\u00", out);
                    fputc(hex[*p >> 4], out);
                    fputc(hex[*p & 0x0f], out);
                } else {
                    fputc(*p, out);
                }
        }
    }
    fputc('"', out);
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static int write_latency_summary(FILE *out, const storage_probe_group_t *group) {
    double values[STORAGE_PROBE_MAX_SAMPLES];
    uint32_t count = 0;
    double sum = 0.0;
    for (uint32_t i = 0; i < group->attempted_samples; ++i) {
        if (group->samples[i].bytes_returned == group->read_size_bytes) {
            values[count++] = group->samples[i].latency_ms;
            sum += group->samples[i].latency_ms;
        }
    }
    if (count == 0) {
        fputs("null", out);
        return 0;
    }
    qsort(values, count, sizeof(values[0]), compare_double);
    const uint32_t p50 = (count + 1) / 2 - 1;
    const uint32_t p95 = (95 * count + 99) / 100 - 1;
    fprintf(out, "{\"unit\":\"ms\",\"percentile_method\":\"nearest_rank\",\"min\":%.6f,\"mean\":%.6f,\"p50\":%.6f,\"p95\":%.6f,\"max\":%.6f}",
            values[0], sum / count, values[p50], values[p95], values[count - 1]);
    return 0;
}

void storage_probe_write_json(FILE *out, const storage_probe_report_t *report) {
    if (!out || !report)
        return;
    fputs("{\"schema_version\":1,\"probe\":\"ps5lm_storageprobe\",\"source_kind\":\"user_selected_regular_file\",\"execution_context\":", out);
    write_json_string(out, report->execution_context[0] ? report->execution_context : "unspecified");
    fputs(",\"file_path\":", out);
    write_json_string(out, report->file_path);
    fprintf(out, ",\"file_size_bytes\":%llu,\"status\":\"%s\",\"error\":{\"code\":\"%s\",\"errno\":%d},",
            (unsigned long long)report->file_size_bytes,
            storage_probe_status_name(report->status), storage_probe_error_name(report->error),
            report->error_number);
    fprintf(out, "\"seed\":%llu,\"limits\":{\"max_logical_read_bytes\":%llu,\"time_limit_ms\":%u,\"samples_per_group\":%u,\"max_concurrent_reads\":%u,\"max_read_buffer_bytes\":%u,\"fixed_report_struct_bytes\":%llu},",
            (unsigned long long)report->seed, (unsigned long long)report->max_logical_bytes,
            report->time_limit_ms, report->samples_per_group, report->max_concurrent_reads,
            report->read_buffer_bytes, (unsigned long long)sizeof(*report));
    fprintf(out, "\"cache_state\":\"unknown\",\"logical_read_bytes\":%llu,\"physical_read_bytes\":null,\"read_accounting\":\"bytes_returned_by_pread\",\"elapsed_ms\":%.6f,\"groups\":[",
            (unsigned long long)report->logical_read_bytes, report->elapsed_ms);

    for (uint32_t i = 0; i < STORAGE_PROBE_READ_SIZE_COUNT; ++i) {
        for (uint32_t p = 0; p < STORAGE_PROBE_PATTERN_COUNT; ++p) {
            const storage_probe_group_t *group = &report->groups[i][p];
            if (i != 0 || p != 0)
                fputc(',', out);
            fprintf(out, "{\"access_pattern\":\"%s\",\"read_size_bytes\":%u,\"status\":\"%s\",\"planned_samples\":%u,\"attempted_samples\":%u,\"completed_samples\":%u,\"logical_read_bytes\":%llu,\"elapsed_ms\":%.6f,\"logical_bytes_per_second\":",
                    p == 0 ? "sequential" : "random", group->read_size_bytes,
                    storage_probe_group_status_name(group->status), group->planned_samples,
                    group->attempted_samples, group->completed_samples,
                    (unsigned long long)group->logical_read_bytes, group->elapsed_ms);
            if (group->elapsed_ms > 0.0) {
                fprintf(out, "%.3f", (double)group->logical_read_bytes * 1000.0 / group->elapsed_ms);
            } else {
                fputs("null", out);
            }
            fputs(",\"latency_summary\":", out);
            write_latency_summary(out, group);
            fputs(",\"samples\":[", out);
            for (uint32_t s = 0; s < group->attempted_samples; ++s) {
                const storage_probe_sample_t *sample = &group->samples[s];
                if (s != 0)
                    fputc(',', out);
                fprintf(out, "{\"offset_bytes\":%llu,\"bytes_returned\":%u,\"latency_ms\":%.6f,\"errno\":%d}",
                        (unsigned long long)sample->offset_bytes, sample->bytes_returned,
                        sample->latency_ms, sample->error_number);
            }
            fputs("]}", out);
        }
    }
    fputs("]}", out);
}
