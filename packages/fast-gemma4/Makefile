CXX        := clang++
SHARED_INC := $(shell if [ -d ../fast-shared/src/fast_shared/include ]; then echo ../fast-shared/src/fast_shared/include; elif [ -d ../../fast-shared/src/fast_shared/include ]; then echo ../../fast-shared/src/fast_shared/include; elif [ -x .venv/bin/python ]; then .venv/bin/python -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; else python3 -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; fi)
PYTHON     := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Explicit deployment target & runtime-safe arm64 target flags:
# - Never pass `-mcpu=apple-m5` blindly (Apple Clang 21.0.0 enables `+specres2` even where FEAT_SPECRES2==0)
#   and avoid `-mcpu=native` (LLVM <23.1 maps M5 to apple-m4).
# - Note: Any local MLX source builds must set CMAKE_OSX_DEPLOYMENT_TARGET >= 26.2 to avoid MLX_METAL_NO_NAX.
# - Core AI (macOS 27.0+) / ahead-of-time Metal Toolchain compilation is optional and runtime-gated.
MACOSX_MIN ?= 15.0
ARCH_FLAGS ?= -arch arm64
CXXFLAGS   := -std=c++17 -O3 $(ARCH_FLAGS) -mmacosx-version-min=$(MACOSX_MIN) -Wall -Wextra -Wno-unused-function -I./native $(if $(SHARED_INC),-I$(SHARED_INC))

ANE_FW     := -framework CoreML -framework Foundation -framework Metal -framework IOSurface -framework CoreVideo -framework Accelerate -framework CoreFoundation

BUILD_DIR  := build

.PHONY: all native guard clean

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/libgemma4_ane_prefix.dylib: native/gemma4_ane_prefix.mm native/gemma4_ane_prefix.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc -shared -fPIC $< $(ANE_FW) -o $@

$(BUILD_DIR)/gemma4-ane-cli: native/gemma4_ane_cli.mm native/gemma4_ane_prefix.mm native/gemma4_ane_prefix.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc native/gemma4_ane_cli.mm native/gemma4_ane_prefix.mm $(ANE_FW) -o $@

native: $(BUILD_DIR)/libgemma4_ane_prefix.dylib $(BUILD_DIR)/gemma4-ane-cli

guard:
	$(PYTHON) -m fast_shared.ir_dtype_guard --strict-bool-mask src/fast_gemma4 tests

clean:
	rm -rf $(BUILD_DIR)
