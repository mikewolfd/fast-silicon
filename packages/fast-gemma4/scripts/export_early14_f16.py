"""Export EarlyExit14 layers `0..14` FP16 CoreML weights (`gemma4_early14_f16.npz`) without `mlx_vlm`."""

from __future__ import annotations

import argparse

from fast_shared.hf import copy_tokenizer_files

from fast_gemma4.model import FastGemma4Model
from fast_gemma4.paths import resolve_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Gemma-4 EarlyExit14 FP16 weights for CoreML compilation")
    parser.add_argument("--home", default=None)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    paths = resolve_paths(args.home)
    model = FastGemma4Model(paths=paths, model_dir=args.model_dir, mlp_dtype="bfloat16", use_mpp=False)
    copy_tokenizer_files(model.model_dir, paths.tokenizer_dir)
    weights_out = paths.resolve_early14_weights_npz(args.model_dir, exit_layer=model.first_kv_shared - 1)
    out = model.export_early14_f16(weights_out)
    print(f"Exported {out} ({out.stat().st_size / (1024**3):.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
