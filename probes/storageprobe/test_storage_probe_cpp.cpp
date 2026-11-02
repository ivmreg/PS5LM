/* Cross-language host check: compile the implementation as C and call it from C++. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L

#include "storage_probe.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <unistd.h>

int main() {
    storage_probe_options_t options{};
    storage_probe_options_default(&options);
    assert(options.samples_per_group > 0);

    char missing_path[128];
    std::snprintf(missing_path, sizeof(missing_path), "/tmp/ps5lm-storageprobe-cxx-%ld", (long)getpid());
    unlink(missing_path);
    storage_probe_report_t report{};
    assert(storage_probe_run(missing_path, &options, nullptr, nullptr, &report) == -1);
    assert(report.error == STORAGE_PROBE_ERROR_OPEN);
    assert(std::strcmp(storage_probe_status_name(report.status), "error") == 0);
    std::puts("storage_probe_cpp_link_test: ok");
    return 0;
}
