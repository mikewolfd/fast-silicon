CXX := clang++
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
SHARED_INC ?= $(shell $(PYTHON) -c "import fast_shared; print(fast_shared.native_include_dir())" 2>/dev/null || echo ../fast-shared/src/fast_shared/include)
MACOSX_DEPLOYMENT_TARGET ?= 15.0
# Keep explicit arm64 target without -mcpu=apple-m5 (Apple Clang 21 enables +specres2 despite
# runtime FEAT_SPECRES=0) or -mcpu=native (LLVM < 23.1 maps M5 to apple-m4).
# Note: MLX source builds separately require CMAKE_OSX_DEPLOYMENT_TARGET >= 26.2 to avoid MLX_METAL_NO_NAX.
ARCH_FLAGS ?= -arch arm64
CXXFLAGS := -std=c++17 -O3 $(ARCH_FLAGS) -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET) -fobjc-arc -Wall -Wextra -I$(SHARED_INC) -Inative
LDFLAGS := -dynamiclib -framework Foundation -framework CoreML -framework CoreVideo
BUILD_DIR := build
TARGET_DYLIB := $(BUILD_DIR)/liblaya_ane_runner.dylib

.PHONY: all native test guard clean

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET_DYLIB): native/laya_ane_runner.mm native/laya_ane_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) native/laya_ane_runner.mm -o $@

native: $(TARGET_DYLIB)

guard:
	@if [ -x .venv/bin/fast-ir-guard ]; then .venv/bin/fast-ir-guard --strict-bool-mask src tests; \
	elif command -v fast-ir-guard >/dev/null 2>&1; then fast-ir-guard --strict-bool-mask src tests; \
	else $(PYTHON) -m fast_shared.ir_dtype_guard --strict-bool-mask src tests; fi

test:
	$(PYTHON) -m pytest -v tests

clean:
	rm -rf $(BUILD_DIR)
