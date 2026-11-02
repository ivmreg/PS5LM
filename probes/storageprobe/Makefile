MODE ?= ps5
BUILD_DIR ?= ../../build/storageprobe
HOST_CC ?= cc
HOST_CXX ?= c++
HOST_CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
HOST_CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -pedantic
COMMON_CFLAGS := -D_POSIX_C_SOURCE=200809L -I.

ifeq ($(MODE),host)

HOST_CFLAGS += $(COMMON_CFLAGS)
HOST_LDLIBS :=
HOST_APP := $(BUILD_DIR)/storageprobe-host
HOST_TEST := $(BUILD_DIR)/storageprobe-test
HOST_CORE_OBJ := $(BUILD_DIR)/storage_probe.host.o
HOST_CPP_TEST_OBJ := $(BUILD_DIR)/storage_probe_cpp_test.o
HOST_CPP_TEST := $(BUILD_DIR)/storageprobe-cxx-link-test

all: host

host: $(HOST_APP)

$(BUILD_DIR):
	mkdir -p "$@"

$(HOST_APP): main.c storage_probe.c storage_probe.h | $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) main.c storage_probe.c -o "$@" $(HOST_LDLIBS)

$(HOST_TEST): test_storage_probe.c storage_probe.c storage_probe.h | $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) test_storage_probe.c storage_probe.c -o "$@" $(HOST_LDLIBS)

$(HOST_CORE_OBJ): storage_probe.c storage_probe.h | $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) -c storage_probe.c -o "$@"

$(HOST_CPP_TEST_OBJ): test_storage_probe_cpp.cpp storage_probe.h | $(BUILD_DIR)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(COMMON_CFLAGS) -c test_storage_probe_cpp.cpp -o "$@"

$(HOST_CPP_TEST): $(HOST_CPP_TEST_OBJ) $(HOST_CORE_OBJ)
	$(HOST_CXX) $^ -o "$@" $(HOST_LDLIBS)

host-test: $(HOST_APP) $(HOST_TEST) $(HOST_CPP_TEST)
	"$(HOST_TEST)"
	"$(HOST_CPP_TEST)"

test: host-test

clean:
	rm -f "$(HOST_APP)" "$(HOST_TEST)" "$(HOST_CORE_OBJ)" "$(HOST_CPP_TEST_OBJ)" "$(HOST_CPP_TEST)"

.PHONY: all host host-test test clean

else

ifdef PS5_PAYLOAD_SDK
    include $(PS5_PAYLOAD_SDK)/toolchain/prospero.mk
else
    $(error PS5_PAYLOAD_SDK is undefined: source scripts/env.sh)
endif

TITLE_OBJECT := $(BUILD_DIR)/storage_probe.title.o

# Compile only the reusable probe core for the Prospero target. The console
# app must link this object through its existing native-title build target.
title-object: $(TITLE_OBJECT)

$(BUILD_DIR):
	mkdir -p "$@"

$(TITLE_OBJECT): storage_probe.c storage_probe.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CFLAGS) -std=c11 -c storage_probe.c -o "$@"

clean:
	rm -f "$(TITLE_OBJECT)"

.PHONY: title-object clean

endif
