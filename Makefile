UV ?= uv
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
NATIVE_PACKAGES := packages/fast-bge-small packages/fast-embeddinggemma-2 packages/fast-gemma4 packages/fast-decision-models packages/fast-pplx

.PHONY: uv-sync hw-check native clean setup guard test-cpu test-hw sync sync-check sync-clean-check leak-check

uv-sync:
	$(UV) sync --extra dev

hw-check:
	$(PYTHON) -c "from fast_shared.hw_check import main; raise SystemExit(main())"

native: hw-check
	@for pkg in $(NATIVE_PACKAGES); do \
		echo "==> Building native targets in $$pkg"; \
		$(MAKE) -C $$pkg PYTHON="$(abspath $(PYTHON))" native || exit 1; \
	done

clean:
	@for pkg in $(NATIVE_PACKAGES); do \
		$(MAKE) -C $$pkg clean; \
	done

SETUP_ARGS ?=

setup:
	$(PYTHON) scripts/setup_all.py $(SETUP_ARGS)

FAST_IR_GUARD ?= $(shell if [ -x .venv/bin/fast-ir-guard ]; then echo .venv/bin/fast-ir-guard; else echo fast-ir-guard; fi)

guard:
	$(FAST_IR_GUARD) --strict-bool-mask --coreml \
		packages/fast-shared/src packages/fast-shared/tests \
		packages/fast-shared/skills/apple-silicon-ml-optimizer/scripts \
		packages/fast-bge-small/src packages/fast-bge-small/tests packages/fast-bge-small/scripts \
		packages/fast-embeddinggemma-2/src packages/fast-embeddinggemma-2/tests packages/fast-embeddinggemma-2/scripts \
		packages/fast-gemma4/src packages/fast-gemma4/tests packages/fast-gemma4/scripts \
		packages/fast-gpt-oss/src packages/fast-gpt-oss/tests \
		packages/fast-pplx/src packages/fast-pplx/tests packages/fast-pplx/scripts \
		packages/fast-decision-models/src packages/fast-decision-models/tests packages/fast-decision-models/scripts \
		scripts tests
	$(PYTHON) -m pytest -q packages/fast-shared/tests/test_ir_dtype_guard_cpu.py -k test_clean_pass_on_fast_repos

test-cpu: hw-check
	$(PYTHON) -m pytest -v \
		packages/fast-shared/tests/test_ir_dtype_guard_cpu.py \
		packages/fast-shared/tests/test_fast_shared.py \
		packages/fast-decision-models/tests/test_decision_models_cpu.py \
		packages/fast-embeddinggemma-2/tests/test_emb2_res_rms_cpu.py \
		packages/fast-gpt-oss/tests/test_fast_gpt_oss.py \
		packages/fast-gemma4/tests/test_per_doc_grammar.py \
		packages/fast-gemma4/tests/test_packager_cpu.py \
		packages/fast-gemma4/tests/test_tier1_cpu.py \
		tests/test_setup_all_cpu.py \
		tests/test_benchmark_suites_cpu.py \
		-k "not mpp_matmul and not _gpu"
	$(PYTHON) -m pytest -v \
		packages/fast-bge-small/tests/test_fast_bge_small.py \
		packages/fast-embeddinggemma-2/tests/test_fast_embeddinggemma_2.py \
		packages/fast-gemma4/tests/test_fast_gemma4.py \
		packages/fast-gemma4/tests/test_ane_hybrid_and_writer.py \
		-k "test_slice_ocr_pages_utf8 or _cpu"
	$(PYTHON) -m pytest -v packages/fast-pplx/tests -m "not gpu_ane"

test-hw: hw-check
	@for pkg in fast-shared fast-bge-small fast-embeddinggemma-2 fast-gpt-oss fast-gemma4 fast-decision-models fast-pplx; do \
		echo "==> Running full pytest suite for packages/$$pkg"; \
		$(PYTHON) -m pytest -v packages/$$pkg/tests || exit 1; \
	done
	$(PYTHON) -m pytest -v tests/test_setup_all_cpu.py tests/test_benchmark_suites_cpu.py

sync:
	$(PYTHON) scripts/sync_from_siblings.py

sync-check:
	$(PYTHON) scripts/sync_from_siblings.py --check

sync-clean-check:
	$(PYTHON) scripts/sync_from_siblings.py --check --require-clean

leak-check:
	@! rg -n -i "/users/|congress|docmeta|hhrg|crpt|senate|committee|appropriations|fused_v4|rosters_lookup" \
		packages scripts tests docs pyproject.toml .agents LICENSE .gitignore uv.lock README.md
