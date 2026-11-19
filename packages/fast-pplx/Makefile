CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter -fPIC -DACCELERATE_NEW_LAPACK
SHARED_INC := $(shell \
  if [ -d ../fast-shared/src/fast_shared/include ]; then \
    echo ../fast-shared/src/fast_shared/include; \
  else \
    .venv/bin/python -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null || \
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
