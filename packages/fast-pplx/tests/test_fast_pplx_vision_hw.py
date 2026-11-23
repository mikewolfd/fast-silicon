"""Real-hardware smoke test for `FastPplx.encode_images`."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
import pytest

from fast_pplx import FastPplx
from fast_pplx.gpu import _find_local_hf_snapshot
from fast_pplx.config import DEFAULT_MODEL_NAME


@pytest.mark.gpu_ane
def test_fast_pplx_encode_images_hardware_smoke() -> None:
    if _find_local_hf_snapshot(DEFAULT_MODEL_NAME) is None:
        pytest.skip(f"Local HF snapshot not staged for {DEFAULT_MODEL_NAME}")

    im = Image.new("RGB", (480, 640), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((24, 30), "ANNUAL INFRASTRUCTURE REPORT", fill=(0, 0, 0))
    dr.text((24, 90), "Municipal water pipeline maintenance budget summary.", fill=(0, 0, 0))

    with FastPplx(mode="gpu", dtype="float16", local_files_only=True) as eng:
        img_vecs = eng.encode_images([im], chunks=4, min_pixels=16384, max_pixels=600_000, normalize=True)
        q_vec = eng.encode_queries(["municipal water pipeline budget"], normalize=True)[0]
        img_i8 = eng.encode_images([im], chunks=1, dimensions=256, output_dtype="int8")[0]

    assert len(img_vecs) == 1
    mat = img_vecs[0]
    assert mat.ndim == 2 and 1 <= mat.shape[0] <= 4 and mat.shape[1] == 1024
    assert np.all(np.isfinite(mat))
    np.testing.assert_allclose(np.linalg.norm(mat, axis=-1), np.ones(mat.shape[0]), atol=1e-3)
    assert float((mat @ q_vec).max()) > 0.15

    assert img_i8.shape == (1, 256)
    assert img_i8.dtype == np.int8
