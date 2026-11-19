CXX      := clang++
SHARED_INC := $(shell if [ -d ../fast-shared/src/fast_shared/include ]; then echo ../fast-shared/src/fast_shared/include; elif [ -d ../../fast-shared/src/fast_shared/include ]; then echo ../../fast-shared/src/fast_shared/include; elif [ -x .venv/bin/python ]; then .venv/bin/python -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; else python3 -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; fi)
MACOS_MIN ?= 15.0
# Portable arm64 baseline: never pass `-mcpu=apple-m5` blindly (Apple Clang 21 enables
# `+specres2` while Darwin reports FEAT_SPECRES2=0, and LLVM <23.1 maps `-mcpu=native`
# on M5 to apple-m4). Any MLX source build requires CMAKE_OSX_DEPLOYMENT_TARGET >= 26.2
# to avoid compiling MLX_METAL_NO_NAX.
ARCH_FLAGS ?= -arch arm64 -mmacosx-version-min=$(MACOS_MIN)
CXXFLAGS := -std=c++17 -O3 $(ARCH_FLAGS) -Wall -Wextra -Wno-unused-function -DACCELERATE_NEW_LAPACK -I./native $(if $(SHARED_INC),-I$(SHARED_INC))

ANE_FW   := -framework CoreML -framework Foundation -framework Accelerate -framework CoreFoundation
CPU_FW   := -framework Accelerate
CLI_FW   := $(ANE_FW) -lz

BUILD_DIR := build

.PHONY: all native cpu-query guard clean

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/libbge_ane_runner.dylib: native/bge_ane_runner.mm native/bge_ane_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc -shared -fPIC $< $(ANE_FW) -o $@

$(BUILD_DIR)/libbge_cpu_query.dylib: native/bge_cpu_query.mm native/bge_cpu_query.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -shared -fPIC $< $(CPU_FW) -o $@

$(BUILD_DIR)/bge-ane-cli: native/bge_ane_cli.mm native/bge_ane_runner.mm native/bge_ane_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc native/bge_ane_cli.mm native/bge_ane_runner.mm $(CLI_FW) -o $@

# Default native build ships the active ANE runner + standalone CLI.
# Optional vectorized CPU query dylib is built via `make cpu-query`.
native: $(BUILD_DIR)/libbge_ane_runner.dylib $(BUILD_DIR)/bge-ane-cli

cpu-query: $(BUILD_DIR)/libbge_cpu_query.dylib

guard:
	.venv/bin/python -m fast_shared.ir_dtype_guard --strict-bool-mask src

clean:
	rm -rf $(BUILD_DIR)
