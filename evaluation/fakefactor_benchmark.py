#!/usr/bin/env python3
"""
Head-to-head: GNN fake factor vs the published binned fake factor, done honestly.

WHY THIS EXISTS
---------------
fakefactor_closure.py compares the two in bins of (tau_eta, tau_prong, tau_pt).
That is the BINNED method's home turf -- it is exact there by construction, so the
GNN can at best tie. Measured that way, with both given the same one-constant
calibration, they are indistinguishable (0.993 vs 0.992 on inclusive yield).

The question that actually matters is different. The binned method ASSUMES

    FF = f(tau_pt, tau_eta, tau_prong)   and nothing else.

If the true fake factor also depends on x, the binned method gets the TOTAL right
while getting the SHAPE wrong -- and the analysis builds its discriminant out of
exactly those other variables, so a wrong shape is a real systematic.

Measured on the run-2 export (binned FF derived on fold_0, calibrated on fold_1,
applied to fold_2), the binned method is biased in variables it does not use:

    mTtau0 : 0.899 0.920 0.988 1.102 1.115   <- monotonic, +-11%
    MET    : 0.975 0.968 0.971 1.000 1.129
    HT     : 1.091 0.965 0.958 0.978 0.936

It cannot fix this: adding a 4th axis multiplies its bins by 5 and destroys its
statistics. The GNN sees all 26 variables at once and pays no such price. THAT is
the claim to test -- not the inclusive yield.

WHAT IT DOES
------------
  * derives the binned FF on --train-data (the fold the GNN trained on)
  * fits ONE global scale for each method on --calib-data (the VALIDATION fold --
    never the apply fold; that would be selecting on the test set)
  * applies both to --data (the apply fold) and compares:
      - inclusive yield
      - bins of tau_pt        (the binned method's home turf: expect a tie)
      - bins of --vars        (the real test: expect the GNN to be flatter)

USAGE
    python evaluation/fakefactor_benchmark.py \
        --pred       <run>/ckpts/<ep>__test_fakeCR.h5   \
        --data       <preproc>/fold_2/fakeCR.h5         \
        --calib-pred <run>/ckpts/<ep>__calib_fold1.h5   \
        --calib-data <preproc>/fold_1/fakeCR.h5         \
        --train-data <preproc>/fold_0/fakeCR.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np

PT_EDGES = np.array([20, 30, 40, 50, 60, 80, 100, 400], float)
DEFAULT_VARS = ["mTtau0", "MET", "HT", "mBB", "dRTauLep", "mTauTauVis"]


def load(path):
    return h5py.File(path, "r")["events"][:]


def get_pID(pred_file, n_expected):
    with h5py.File(pred_file, "r") as f:
        for key in f:
            ds = f[key]
            if ds.dtype.names is None:
                continue
            pid = next((n for n in ds.dtype.names if n.endswith("pID")), None)
            pan = next((n for n in ds.dtype.names if n.endswith("pantiID")), None)
            if pid and pan:
                arr = ds[:]
                d = arr[pid].astype(float) + arr[pan].astype(float)
                p = np.where(d > 0, arr[pid].astype(float) / d, 0.0)
                if len(p) != n_expected:
                    sys.exit(f"ERROR: {pred_file} has {len(p):,} rows, data has {n_expected:,}")
                return p
    sys.exit(f"ERROR: no pID/pantiID in {pred_file}")


def keys_of(ev):
    pt = np.clip(ev["tau_pt"], PT_EDGES[0], PT_EDGES[-1] - 1e-3)
    return ev["is_endcap"].astype(int), ev["tau_prong"].astype(int), np.digitize(pt, PT_EDGES) - 1


def derive_binned(ev):
    """The published method: FF = sum(w)_ID / sum(w)_antiID per (eta, prong, pT)."""
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"]
    e, pr, b = keys_of(ev)
    ff = {}
    for ee in (0, 1):
        for pp in (1, 3):
            for bb in range(len(PT_EDGES) - 1):
                c = (e == ee) & (pr == pp) & (b == bb)
                A = w[c & (idf == 1)].sum()
                B = w[c & (idf == 0)].sum()
                # make_positive(): the published method clamps unphysical bins
                ff[(ee, pp, bb)] = max(A / B, 0.0) if B > 0 else 0.0
    return ff


def binned_ff_of(ev, ffmap):
    e, pr, b = keys_of(ev)
    return np.array([ffmap.get((a, c, d), 0.0) for a, c, d in zip(e, pr, b)])


def scale_on(ev, ff):
    """One global constant, fitted where the model is neither trained nor applied."""
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"]
    pred = (w[idf == 0] * ff[idf == 0]).sum()
    return w[idf == 1].sum() / pred if pred else 1.0


def table(ev, ffs, names, var, edges=None, label=None):
    """Predicted/actual ratio per bin of `var`, for each method in ffs."""
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"]
    x = ev[var] if var != "tau_pt_binned" else ev["tau_pt"]
    if edges is None:
        edges = np.percentile(x, [0, 20, 40, 60, 80, 100])
    print(f"\n  --- {label or var} ---")
    print(f"  {'bin':>20}{'actual':>10}" + "".join(f"{n:>12}" for n in names))
    devs = {n: [] for n in names}
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & ((x < edges[i + 1]) if i < len(edges) - 2 else (x <= edges[i + 1]))
        act = w[m & (idf == 1)].sum()
        if act <= 0:
            continue
        row = f"  {edges[i]:>9.0f}-{edges[i+1]:<10.0f}{act:>10,.0f}"
        for n, ff in zip(names, ffs):
            pred = (w[m & (idf == 0)] * ff[m & (idf == 0)]).sum()
            r = pred / act
            devs[n].append(r)
            row += f"{r:>12.3f}"
        print(row)
    print(f"  {'RMS deviation':>20}{'':>10}" + "".join(
        f"{np.sqrt(np.mean((np.array(devs[n])-1)**2)):>12.3f}" for n in names))
    return {n: np.sqrt(np.mean((np.array(devs[n]) - 1) ** 2)) for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="GNN prediction on the APPLY fold")
    ap.add_argument("--data", required=True, help="the APPLY fold fakeCR.h5")
    ap.add_argument("--calib-pred", required=True, help="GNN prediction on the VALIDATION fold")
    ap.add_argument("--calib-data", required=True, help="the VALIDATION fold fakeCR.h5")
    ap.add_argument("--train-data", required=True, help="the TRAIN fold, to derive the binned FF")
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    ap.add_argument("--no-calib", action="store_true", help="skip the global calibration")
    args = ap.parse_args()

    te, va, tr = load(args.data), load(args.calib_data), load(args.train_data)
    print(f"train {len(tr):,}   calib {len(va):,}   apply {len(te):,}")

    ffmap = derive_binned(tr)
    b_te, b_va = binned_ff_of(te, ffmap), binned_ff_of(va, ffmap)

    p_te = get_pID(args.pred, len(te))
    p_va = get_pID(args.calib_pred, len(va))
    g_te = np.clip(p_te, 1e-6, 1 - 1e-6)
    g_te = g_te / (1 - g_te)
    g_va = np.clip(p_va, 1e-6, 1 - 1e-6)
    g_va = g_va / (1 - g_va)

    if not args.no_calib:
        cb, cg = scale_on(va, b_va), scale_on(va, g_va)
        print(f"global scale from the validation fold:  binned {cb:.4f}   GNN {cg:.4f}")
        print("  (one constant each, fitted off the apply fold -- shapes untouched)")
        b_te, g_te = b_te * cb, g_te * cg

    w = te["fakefactor_weight"].astype(float)
    idf = te["is_idfake"]
    act = w[idf == 1].sum()
    print(f"\n{'INCLUSIVE':>20}{'actual':>10}{'binned':>12}{'GNN':>12}")
    row = f"  {'':>18}{act:>10,.0f}"
    for ff in (b_te, g_te):
        row += f"{(w[idf==0]*ff[idf==0]).sum()/act:>12.3f}"
    print(row)

    names = ["binned", "GNN"]
    print("\n" + "=" * 72)
    print("HOME TURF of the binned method -- it is exact here by construction.")
    print("=" * 72)
    table(te, [b_te, g_te], names, "tau_pt", edges=PT_EDGES, label="tau_pt (the binned FF's own axis)")

    print("\n" + "=" * 72)
    print("THE REAL TEST -- variables the binned FF does NOT use.")
    print("A flat column of 1.000 means the method models the fake factor's")
    print("dependence on that variable. The binned method cannot: adding an axis")
    print("would multiply its bins and destroy its statistics.")
    print("=" * 72)
    wins = {"binned": 0, "GNN": 0}
    for v in args.vars:
        if v not in te.dtype.names:
            print(f"\n  (skipping {v}: not in the h5)")
            continue
        rms = table(te, [b_te, g_te], names, v)
        wins["GNN" if rms["GNN"] < rms["binned"] else "binned"] += 1
    print("\n" + "=" * 72)
    print(f"VERDICT over {sum(wins.values())} unused variables: "
          f"GNN flatter in {wins['GNN']}, binned flatter in {wins['binned']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
