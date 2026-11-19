CXX        := clang++
SHARED_INC := $(shell if [ -d ../fast-shared/src/fast_shared/include ]; then echo ../fast-shared/src/fast_shared/include; elif [ -d ../../fast-shared/src/fast_shared/include ]; then echo ../../fast-shared/src/fast_shared/include; elif [ -x .venv/bin/python ]; then .venv/bin/python -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; else python3 -c "import fast_shared, pathlib; print(pathlib.Path(fast_shared.__file__).resolve().parent / 'include')" 2>/dev/null; fi)
CXXFLAGS   := -std=c++17 -O3 -Wall -Wextra -Wno-unused-function -I./native $(if $(SHARED_INC),-I$(SHARED_INC))

ANE_FW     := -framework CoreML -framework Foundation -framework Metal -framework IOSurface -framework CoreVideo -framework Accelerate -framework CoreFoundation

BUILD_DIR  := build

.PHONY: all native clean

all: native

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/libgemma4_ane_prefix.dylib: native/gemma4_ane_prefix.mm native/gemma4_ane_prefix.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc -shared -fPIC $< $(ANE_FW) -o $@

$(BUILD_DIR)/gemma4-ane-cli: native/gemma4_ane_cli.mm native/gemma4_ane_prefix.mm native/gemma4_ane_prefix.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -ObjC++ -fobjc-arc native/gemma4_ane_cli.mm native/gemma4_ane_prefix.mm $(ANE_FW) -o $@

native: $(BUILD_DIR)/libgemma4_ane_prefix.dylib $(BUILD_DIR)/gemma4-ane-cli

clean:
	rm -rf $(BUILD_DIR)
