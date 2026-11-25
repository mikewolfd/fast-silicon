# Contributing to `fast-silicon`

## Development setup

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --extra dev --extra extract --extra build
```

Preview required weights and Core ML bundles before downloading (~36-38 GB full across all 8 model packages, or ~25.6 GB with `--skip-test-extras`):

```bash
.venv/bin/python scripts/setup_all.py --dry-run
make setup SETUP_ARGS="--packages bge-small,pplx"
```

## Verification and test targets

Hardware suites touch the GPU and Apple Neural Engine and self-serialize on a host lockfile (`/tmp/m4_gpu_validation.lock`) so concurrent processes do not contend on accelerators.

```bash
make test-cpu   # Fast CPU-only regression tests across all packages
make test-hw    # Full per-package test suites (GPU / ANE serialized)
make guard      # Static dtype, boolean-mask, and Core ML placement lints
make native     # Rebuild native C++/Objective-C++ ANE & NEON libraries
```

## Monorepo layout and sibling synchronization

Six packages (`fast-shared`, `fast-bge-small`, `fast-embeddinggemma-2`, `fast-gpt-oss`, `fast-gemma4`, `fast-decision-models`) mirror standalone sibling checkouts when developed locally, while `fast-pplx`, `fast-qwen3`, and `fast-ocr` live only in this repository.

```bash
make sync             # Copy clean sources from sibling ../fast-* checkouts
make sync-check       # Verify monorepo copies match sibling checkouts
make sync-clean-check # Also require clean git working trees in siblings
make leak-check       # Scan tracked files for local paths or private strings
```

## Benchmarks

For cross-silicon methodology, dataset definitions, and reproducible CLI invocations, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md#8-reproducible-cli-invocations). Hardware dispatch policies live in [runtime_lessons.py](packages/fast-shared/src/fast_shared/runtime_lessons.py).
