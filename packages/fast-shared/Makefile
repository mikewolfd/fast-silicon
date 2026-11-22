PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif [ -x ../../.venv/bin/python ]; then echo ../../.venv/bin/python; else echo python3; fi)

.PHONY: all native guard clean

all: native

native:
	$(MAKE) -C native_ane all

guard:
	$(PYTHON) -m fast_shared.ir_dtype_guard --strict-bool-mask --coreml src tests

clean:
	$(MAKE) -C native_ane clean
