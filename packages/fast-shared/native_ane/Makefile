CXX ?= clang++
SDKROOT := $(shell xcrun --show-sdk-path 2>/dev/null)

CXXFLAGS := -std=c++17 -O3 -arch arm64 -mmacosx-version-min=15.0 \
	-Wall -Wextra -Werror -ObjC++ -fobjc-arc -fPIC
ifneq ($(SDKROOT),)
CXXFLAGS += -isysroot $(SDKROOT)
endif

LDFLAGS := -shared -arch arm64 -mmacosx-version-min=15.0 \
	-framework Foundation -framework Metal -framework IOSurface -framework CoreFoundation

BUILD_DIR := build
TARGET := $(BUILD_DIR)/libfast_ane_handoff.dylib
SRCS := fast_ane_handoff.mm
HDRS := fast_ane_handoff.h

.PHONY: all native clean

all: $(TARGET)

native: all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRCS) $(HDRS) Makefile | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -rf $(BUILD_DIR)
