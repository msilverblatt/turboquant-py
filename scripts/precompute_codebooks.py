"""Precompute and ship codebook arrays for common (dim, bit_width) combinations.

Saves each codebook as a .npz file in src/turboquant/codebooks/ so that
get_codebook() can load them directly instead of computing on-the-fly.
"""

from __future__ import annotations

import pathlib

import numpy as np

from turboquant.codebook import compute_codebook

BIT_WIDTHS = [1, 2, 3, 4]
DIMS = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

CODEBOOKS_DIR = pathlib.Path(__file__).parent.parent / "src" / "turboquant" / "codebooks"


def main() -> None:
    CODEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(DIMS) * len(BIT_WIDTHS)
    done = 0
    for dim in DIMS:
        for bw in BIT_WIDTHS:
            centroids, boundaries = compute_codebook(dim, bw)
            out_path = CODEBOOKS_DIR / f"codebook_dim{dim}_bw{bw}.npz"
            np.savez(out_path, centroids=centroids, boundaries=boundaries)
            done += 1
            print(f"[{done}/{total}] Saved {out_path.name}")
    print(f"Done. {done} codebook files written to {CODEBOOKS_DIR}")


if __name__ == "__main__":
    main()
