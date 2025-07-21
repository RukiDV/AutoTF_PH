#!/usr/bin/env python3
"""
evaluate_persistence.py

Compute J_arc, J_box, Precision and Recall for each persistence pair
by back‐projecting its voxels into the (scalar,gradient) TF plane.
"""

import numpy as np
import pandas as pd
import argparse

def clamp_and_expand(birth, death, B):
    # sort & clamp
    bs, ds = sorted((birth, death))
    bs = max(0, min(bs, B-1))
    ds = max(0, min(ds, B-1))
    # pad zero‐width features so they cover at least two bins
    if bs == ds:
        bs = max(0, bs-1)
        ds = min(B-1, ds+1)
    return bs, ds

def compute_metrics(vol, grad, bs, ds, B):
    # compute the persistent‐feature mask P
    maskP = (vol >= bs) & (vol <= ds)

    # flip gradient for plotting/bin‐coords
    fg = (B - 1) - grad

    # find the tight bounding‐box of P in (s,fg)‐space
    fg_vals = fg[maskP]
    if fg_vals.size == 0:
        # no voxels ⇒ all metrics zero
        return 0.0, 0.0, 0.0, 0.0
    gmin, gmax = int(fg_vals.min()), int(fg_vals.max())

    # build the axis‐aligned “box” mask
    maskA = (vol >= bs) & (vol <= ds) & (fg >= gmin) & (fg <= gmax)

    # compute TP, and the raw counts
    TP       = np.logical_and(maskP, maskA).sum()
    countP   = maskP.sum()
    countBox = maskA.sum()
    union    = np.logical_or(maskP, maskA).sum()

    # now the four metrics
    J_arc     = TP / union              if union     > 0 else 0.0
    J_box     = TP / countBox           if countBox  > 0 else 0.0
    precision = TP / countP             if countP    > 0 else 0.0
    recall    = TP / countBox           if countBox  > 0 else 0.0

    return J_arc, J_box, precision, recall

def main():
    p = argparse.ArgumentParser(
      description="Evaluate persistence reprojection metrics"
    )
    p.add_argument("--volume",   required=True,
                   help="path to scalar_volume.bin (uint8)")
    p.add_argument("--gradient", required=True,
                   help="path to gradient_volume.bin (uint8)")
    p.add_argument("--pairs",    required=True,
                   help="CSV (birth,death) of persistence pairs")
    p.add_argument("--bins",     type=int, default=256,
                   help="number of TF bins (default: 256)")
    p.add_argument("--output",   default="metrics.csv",
                   help="where to write per‐pair metrics")
    args = p.parse_args()

    # load volumes
    vol = np.fromfile(args.volume,   dtype=np.uint8)
    grd = np.fromfile(args.gradient, dtype=np.uint8)
    assert vol.shape == grd.shape, "volume/gradient size mismatch"

    # load persistence pairs
    df = pd.read_csv(args.pairs)
    assert {"birth","death"}.issubset(df.columns)

    # for each pair compute metrics
    records = []
    for idx, row in df.iterrows():
        b,d = int(row.birth), int(row.death)
        bs, ds = clamp_and_expand(b, d, args.bins)
        J_arc, J_box, prec, rec = compute_metrics(vol, grd, bs, ds, args.bins)
        records.append({
          "birth": b,
          "death": d,
          "J_arc": J_arc,
          "J_box": J_box,
          "precision": prec,
          "recall": rec
        })

    out = pd.DataFrame.from_records(records,
        columns=["birth","death","J_arc","J_box","precision","recall"])
    out.to_csv(args.output, index=False)
    print(f"Wrote metrics for {len(out)} pairs to {args.output}")

if __name__=="__main__":
    main()
