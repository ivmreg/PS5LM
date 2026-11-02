/* Host command-line frontend for storage_probe.c. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L

#include "storage_probe.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MiB (1024ull * 1024ull)

static volatile sig_atomic_t g_interrupted;

static void handle_signal(int signal_number) {
    g_interrupted = signal_number;
}

static int is_cancelled(void *context) {
    (void)context;
    return g_interrupted != 0;
}

static int parse_u64(const char *text, uint64_t *out) {
    if (!text || !*text || *text == '-')
        return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !end || *end != '\0')
        return 0;
    *out = (uint64_t)value;
    return 1;
}

static void usage(FILE *out, const char *program) {
    fprintf(out,
            "Usage: %s --file PATH [--seed N] [--samples N] [--byte-limit-mib N] [--time-limit-sec N]\n"
            "Reads a user-selected existing regular file with one synchronous read at a time.\n"
            "Limits: samples 1..%u; logical bytes 1..80 MiB; time 1..60 seconds.\n",
            program, STORAGE_PROBE_MAX_SAMPLES);
}

int main(int argc, char **argv) {
    storage_probe_options_t options;
    storage_probe_options_default(&options);
    const char *file_path = NULL;
    int want_help = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            want_help = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
        const char *value = argv[++i];
        uint64_t parsed = 0;
        if (strcmp(argv[i - 1], "--file") == 0) {
            if (file_path) {
                fputs("--file may be supplied only once\n", stderr);
                return 2;
            }
            file_path = value;
        } else if (strcmp(argv[i - 1], "--seed") == 0) {
            if (!parse_u64(value, &options.seed)) {
                fputs("invalid unsigned --seed value\n", stderr);
                return 2;
            }
        } else if (strcmp(argv[i - 1], "--samples") == 0) {
            if (!parse_u64(value, &parsed) || parsed == 0 || parsed > STORAGE_PROBE_MAX_SAMPLES) {
                fputs("--samples must be in 1..32\n", stderr);
                return 2;
            }
            options.samples_per_group = (uint32_t)parsed;
        } else if (strcmp(argv[i - 1], "--byte-limit-mib") == 0) {
            if (!parse_u64(value, &parsed) || parsed == 0 || parsed > 80) {
                fputs("--byte-limit-mib must be in 1..80\n", stderr);
                return 2;
            }
            options.max_logical_bytes = parsed * MiB;
        } else if (strcmp(argv[i - 1], "--time-limit-sec") == 0) {
            if (!parse_u64(value, &parsed) || parsed == 0 || parsed > 60) {
                fputs("--time-limit-sec must be in 1..60\n", stderr);
                return 2;
            }
            options.time_limit_ms = (uint32_t)(parsed * 1000);
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i - 1]);
            usage(stderr, argv[0]);
            return 2;
        }
    }
    if (want_help) {
        usage(stdout, argv[0]);
        return 0;
    }
    if (!file_path) {
        fputs("--file is required\n", stderr);
        usage(stderr, argv[0]);
        return 2;
    }

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);

    storage_probe_report_t report;
    const int result = storage_probe_run(file_path, &options, is_cancelled, NULL, &report);
    snprintf(report.execution_context, sizeof(report.execution_context), "host");
    storage_probe_write_json(stdout, &report);
    fputc('\n', stdout);
    fflush(stdout);
    if (result == 0)
        return 0;
    if (result == 1)
        return g_interrupted ? 128 + g_interrupted : 130;
    if (result == 2)
        return 124;
    return 1;
}
