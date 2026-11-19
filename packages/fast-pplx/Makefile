# Native C++/ObjC++ build for fast-pplx (Accelerate/NEON + CoreML staged ANE runner).
#
# Deployment & architecture notes:
# - Default `-mcpu=apple-m4` is safe across M4 (Apple9) and M5+ (Apple10+) Macs.
# - Never pass blind `-mcpu=apple-m5`: Apple Clang 21 enables `+specres2` even though
#   `FEAT_SPECRES`/`FEAT_SPECRES2` report 0 at runtime (guarded below with `+nospecres2`).
# - For MLX source builds targeting M5 NAX, set `CMAKE_OSX_DEPLOYMENT_TARGET>=26.2`
#   to prevent compiling `MLX_METAL_NO_NAX`.
# - Optional macOS 27.0+ Core AI builds should gate explicitly on `USE_OS_COREAI=1`
#   and `MACOSX_MIN_VERSION>=27.0`.

CXX := clang++
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
MACOSX_MIN_VERSION ?= 15.0
TARGET_ARCH ?= arm64
TARGET_CPU ?= apple-m4
CPU_FLAGS := $(if $(filter apple-m5,$(TARGET_CPU)),-mcpu=apple-m5+nospecres2,$(if $(TARGET_CPU),-mcpu=$(TARGET_CPU),))

CXXFLAGS := -std=c++17 -O3 -arch $(TARGET_ARCH) -mmacosx-version-min=$(MACOSX_MIN_VERSION) $(CPU_FLAGS) \
  -Wall -Wextra -Wno-unused-parameter -fPIC -DACCELERATE_NEW_LAPACK
SHARED_INC := $(shell \
  if [ -d ../fast-shared/src/fast_shared/include ]; then \
    echo ../fast-shared/src/fast_shared/include; \
  else \
    $(PYTHON) -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null || \
    python3 -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')"; \
  fi)
OBJCXX_FLAGS := -ObjC++ -fobjc-arc -I$(SHARED_INC)
FRAMEWORKS := -framework CoreML -framework Foundation -framework Accelerate -framework CoreFoundation
ZLIB := -lz

BUILD_DIR := build
NATIVE_DIR := native

TARGETS := \
  $(BUILD_DIR)/libpplx_fast_io.dylib \
  $(BUILD_DIR)/libpplx_coreml_runner.dylib \
  $(BUILD_DIR)/pplx-ane-cli

.PHONY: native all clean

native: $(TARGETS)

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/libpplx_fast_io.dylib: $(NATIVE_DIR)/pplx_fast_io.cpp $(NATIVE_DIR)/pplx_fast_io.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SHARED_INC) -dynamiclib \
	  -install_name @loader_path/libpplx_fast_io.dylib \
	  $< -o $@ -framework Accelerate

$(BUILD_DIR)/libpplx_coreml_runner.dylib: $(NATIVE_DIR)/pplx_coreml_runner.mm $(NATIVE_DIR)/pplx_coreml_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJCXX_FLAGS) -dynamiclib \
	  -install_name @loader_path/libpplx_coreml_runner.dylib \
	  $< -o $@ $(FRAMEWORKS)

$(BUILD_DIR)/pplx-ane-cli: $(NATIVE_DIR)/pplx_ane_cli.mm $(NATIVE_DIR)/pplx_fast_io.cpp $(NATIVE_DIR)/pplx_coreml_runner.mm $(NATIVE_DIR)/pplx_fast_io.h $(NATIVE_DIR)/pplx_coreml_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJCXX_FLAGS) \
	  $(NATIVE_DIR)/pplx_ane_cli.mm $(NATIVE_DIR)/pplx_fast_io.cpp $(NATIVE_DIR)/pplx_coreml_runner.mm \
	  -o $@ $(FRAMEWORKS) $(ZLIB)

clean:
	rm -rf $(BUILD_DIR)
