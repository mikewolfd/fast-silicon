CXX        := clang++
SHARED_INC := $(shell if [ -d ../fast-shared/src/fast_shared/include ]; then echo ../fast-shared/src/fast_shared/include; elif [ -d ../../fast-shared/src/fast_shared/include ]; then echo ../../fast-shared/src/fast_shared/include; elif [ -x .venv/bin/python ]; then .venv/bin/python -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; else python3 -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; fi)
CXXFLAGS   := -std=c++17 -O3 -Wall -Wextra -Wno-unused-function -DACCELERATE_NEW_LAPACK -I./native $(if $(SHARED_INC),-I$(SHARED_INC))

ANE_FW     := -framework CoreML -framework Foundation -framework Accelerate -framework CoreFoundation
IO_FW      := -framework Accelerate
CLI_FW     := $(ANE_FW) -lz

BUILD_DIR  := build

.PHONY: all native clean

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/libemb2_coreml_runner.dylib: native/emb2_coreml_runner.mm native/emb2_coreml_runner.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc -shared -fPIC $< $(ANE_FW) -o $@

$(BUILD_DIR)/libemb2_fast_io.dylib: native/emb2_fast_io.cpp native/emb2_fast_io.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC $< $(IO_FW) -o $@

$(BUILD_DIR)/emb2-ane-cli: native/emb2_ane_cli.mm native/emb2_coreml_runner.mm native/emb2_fast_io.cpp native/emb2_coreml_runner.h native/emb2_fast_io.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc native/emb2_ane_cli.mm native/emb2_coreml_runner.mm native/emb2_fast_io.cpp $(CLI_FW) -o $@

native: $(BUILD_DIR)/libemb2_coreml_runner.dylib $(BUILD_DIR)/libemb2_fast_io.dylib $(BUILD_DIR)/emb2-ane-cli

clean:
	rm -rf $(BUILD_DIR)
