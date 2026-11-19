#!/usr/bin/env python3
"""Export BGE-small-en-v1.5 weights (FP16 .npz) and tokenizer files into artifacts/."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir

DEFAULT_REPO_ID = "mlx-community/bge-small-en-v1.5-bf16"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def export_from_dir(model_dir: Path, home_dir: Path) -> Path:
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {model_dir}")

    weights_dir = home_dir / "artifacts" / "weights"
    tok_dir = home_dir / "artifacts" / "tokenizer"
    weights_dir.mkdir(parents=True, exist_ok=True)

    raw_mx = mx.load(str(safetensors_path))
    np_f16: dict[str, np.ndarray] = {}
    for key, arr in sorted(raw_mx.items()):
        if key.startswith("pooler.") or "position_ids" in key:
            continue
        np_f16[key] = np.array(arr.astype(mx.float16), dtype=np.float16)

    out_npz = weights_dir / "bge_small_f16.npz"
    np.savez(out_npz, **np_f16)
    copy_tokenizer_files(model_dir, tok_dir, TOKENIZER_FILES)

    print(f"Exported {len(np_f16)} tensors to {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB)")
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BGE-small weights and tokenizer assets.")
    parser.add_argument("--home", type=Path, default=repo_root(), help="fast-bge-small root directory")
    parser.add_argument("--model-dir", type=Path, default=None, help="Local HF snapshot directory containing model.safetensors")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace repo ID if --model-dir is omitted")
    parser.add_argument("--local-files-only", action="store_true", help="Do not hit network during snapshot resolution")
    args = parser.parse_args()

    model_dir = resolve_hf_model_dir(args.model_dir, args.repo_id, local_files_only=args.local_files_only)
    export_from_dir(model_dir, args.home.resolve())


if __name__ == "__main__":
    main()
