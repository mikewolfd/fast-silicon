"""Build and verify the lossless English+JSON+legal vocab remap (`262,144 -> 150,786`) for `fast-gemma4`."""

from __future__ import annotations

import argparse

from fast_gemma4.paths import resolve_paths
from fast_gemma4.tokenizer import DEFAULT_KEPT_VOCAB_SIZE, FastGemma4Tokenizer, VocabRemapper


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lossless Gemma-4 vocab remap table")
    parser.add_argument("--home", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--target-size", type=int, default=DEFAULT_KEPT_VOCAB_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = resolve_paths(args.home)
    tok_json = paths.resolve_tokenizer_json(args.model_dir, local_files_only=True)
    if args.force and paths.vocab_remap_npz.exists():
        paths.vocab_remap_npz.unlink()

    rem = VocabRemapper.load_or_build(tok_json, paths.vocab_remap_npz, target_size=args.target_size)
    tok = FastGemma4Tokenizer(tok_json)
    probe = (
        '{"status":"ok","primary_kind":"technical_report","domain":"energy","date":"2026-11-15",'
        '"entities":["Pacific Transmission Consortium","IEEE-2800-2022 — §101"]}'
    )
    old_ids = tok.encode(probe, add_special_tokens=False)
    new_ids, misses = rem.encode_remap(old_ids)
    decoded = tok.decode(rem.decode_to_old(new_ids), skip_special_tokens=False)
    print(f"Saved {paths.vocab_remap_npz} (vocab_out={rem.vocab_size}, probe_misses={misses}, exact={decoded == probe})")
    if decoded != probe:
        raise RuntimeError("Lossless vocab remap verification failed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
