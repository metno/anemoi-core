#!/usr/bin/env python
# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gaussian smoothing matrices for the multiscale losses, built row-chunk by row-chunk.

    w_ij = exp(-d_ij² / (2 σ²)),   d = great-circle angle [rad],   |d| <= cutoff · σ,   rows l1-normalised,

saved as a scipy CSR ``.npz`` of shape (n_target, n_source) — the file mode of
``anemoi.training.losses.MultiscaleLossWrapper`` and of ``anemoi.training.spread``. The node
coordinates and their ORDER are read from the training graph's data nodes, so the matrix matches
the model grid.

Scale convention (as the on-the-fly smoothers ``smooth_<k>x``): σ = k · h with h the area-equivalent
mean grid spacing sqrt(4π / n). Give each smoother as  name:k:cutoff_in_sigmas  (k in units of h)
or  name:<sigma>rad:cutoff  for an absolute σ. List the matrices COARSEST FIRST in the loss config.

Cost: the non-zeros per row grow with (cutoff · k)²; σ = 8 h with cutoff 2 σ on an N320 grid
(542 080 nodes) gives 4.4e8 non-zeros, a 3.5 GB file, about 10 GB of memory and a few minutes.

    python tools/spread_calibration/build_smoothers.py --graph graph.pt --out smoothers \
        --smoother gauss_8x:8:2.0 --smoother gauss_2x:2:2.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree


def xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def build(p: np.ndarray, sigma: float, cutoff_sigmas: float, chunk: int) -> tuple[sp.csr_matrix, dict]:
    n = p.shape[0]
    tree = cKDTree(p)
    r_chord = 2.0 * np.sin(min(cutoff_sigmas * sigma, np.pi) / 2.0)
    indptr = np.zeros(n + 1, dtype=np.int64)
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    d2w_sum = 0.0  # sum over rows of sum_j w_ij d_ij² (normalised weights) -> effective width
    t0 = time.time()
    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        nbl = tree.query_ball_point(p[rows], r_chord, workers=-1)  # includes the node itself (d = 0, w = 1)
        lens = np.fromiter((len(x) for x in nbl), dtype=np.int64, count=len(nbl))
        c = np.concatenate([np.asarray(x, dtype=np.int64) for x in nbl])
        ri = np.repeat(rows, lens)
        d = 2.0 * np.arcsin(np.clip(np.linalg.norm(p[c] - p[ri], axis=1) / 2.0, 0.0, 1.0))
        w = np.exp(-d * d / (2.0 * sigma * sigma))
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        w /= np.repeat(np.add.reduceat(w, starts), lens)
        d2w_sum += float(np.sum(w * d * d))
        # canonical CSR: columns sorted within each row
        order = np.lexsort((c, ri))
        cols.append(c[order].astype(np.int32))
        vals.append(w[order].astype(np.float32))
        indptr[rows + 1] = lens
    indptr = np.cumsum(indptr)
    m = sp.csr_matrix((np.concatenate(vals), np.concatenate(cols), indptr), shape=(n, n))
    nnz_row = np.diff(m.indptr)
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    h = float(np.sqrt(4.0 * np.pi / n))
    meta = {
        "sigma_rad": sigma,
        "sigma_km": float(sigma * 6371.0),
        "sigma_over_h": float(sigma / h),
        "h_rad": h,
        "cutoff_sigmas": cutoff_sigmas,
        "n_nodes": int(n),
        "nnz": int(m.nnz),
        "neighbours_per_row_mean": float(nnz_row.mean()),
        "neighbours_per_row_min": int(nnz_row.min()),
        "neighbours_per_row_max": int(nnz_row.max()),
        "sigma_eff_over_sigma": float(np.sqrt(d2w_sum / n / 2.0) / sigma),  # 2-D Gaussian: <d²> = 2 σ²
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "build_seconds": time.time() - t0,
        "weight_formula": "exp(-d^2/(2 sigma^2)), d = great-circle angle [rad], l1 row-normalised, self weight included",
    }
    return m, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="anemoi graph .pt (HeteroData) with the data nodes")
    ap.add_argument("--node-name", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoother", action="append", required=True, help="name:<k | <sigma>rad>:cutoff_sigmas (repeatable)")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()
    graph = torch.load(a.graph, weights_only=False, map_location="cpu")
    coords = graph[a.node_name].x.numpy().astype(np.float64)  # (N, 2) [lat, lon] in radians
    n = coords.shape[0]
    h = float(np.sqrt(4.0 * np.pi / n))
    print(f"graph {a.graph}: {n} {a.node_name} nodes, h = {h:.6f} rad = {h * 6371.0:.1f} km", flush=True)
    p = xyz(coords[:, 0], coords[:, 1])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.smoother:
        name, scale, cutoff = spec.split(":")
        sigma = float(scale[:-3]) if scale.endswith("rad") else float(scale) * h
        m, meta = build(p, sigma, float(cutoff), a.chunk)
        sp.save_npz(out / f"{name}.npz", m, compressed=False)
        meta.update(file=f"{name}.npz", graph=str(a.graph), spec=spec)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(
            f"{name}: sigma = {sigma:.6f} rad = {meta['sigma_over_h']:.2f} h = {meta['sigma_km']:.0f} km, cutoff {cutoff} sigma, "
            f"nnz {meta['nnz']:,} ({meta['neighbours_per_row_mean']:.0f}/row), sigma_eff/sigma {meta['sigma_eff_over_sigma']:.3f}, "
            f"row sums [{meta['row_sum_min']:.6f}, {meta['row_sum_max']:.6f}], {meta['build_seconds']:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
