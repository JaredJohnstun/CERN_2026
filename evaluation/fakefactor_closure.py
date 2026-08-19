#!/usr/bin/env python3
"""
Fake-factor closure: does the GNN reproduce (and smooth) the binned fake factor?

For each (Barrel/Endcap x 1-/3-prong x Tau1Pt bin) we compare:

  * binned FF   = sum(w)_ID / sum(w)_antiID           (the published method)
                  with w = fakefactor_weight (data - non-veto MC), split by is_idfake
  * GNN FF      = < p_ID / (1 - p_ID) >  averaged over ANTI-ID DATA events in the bin
                  i.e. the per-event fake factor the network would apply to anti-ID data

If the GNN learned the fake factor, the per-bin average of its smooth FF(x) tracks
the binned FF -- while also varying continuously within each bin.

Two modes:
  real     --pred <salt_prediction.h5> --data <fakeCR.h5>
             reads p_ID from the SALT PredictionWriter output (columns 'pID'/'pantiID')
  --standin-demo --data <fakeCR.h5>
             trains a quick gradient-boosted stand-in on DATA-only ID-vs-antiID to
             validate the whole closure loop locally (no GPU / no SALT needed). In this
             mode the binned FF is the RAW data TL/TA ratio (no subtraction), because
             sklearn cannot use the signed subtraction weight; it demonstrates the
             mechanism "classifier odds == TL/TA ratio". SALT handles the signed
             subtraction itself via sample_weight.

Published Tau1Pt edges from HHARD/scripts/bbtt_fake_factors_LH.py.
Barrel = |tau_eta| < 1.5 (bbttSelections.py).
"""
import argparse
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT_EDGES = np.array([20, 30, 40, 50, 60, 80, 100, 400], float)

# event-level features used by the stand-in (must exist in the fakeCR events dataset)
STANDIN_FEATURES = [
    "pTHH", "etaHH", "mHH", "mBB", "MET", "mjjVBF", "dRjjVBF", "Eta0Eta1VBF",
    "dRb0tau", "dRb1tau", "dRBB", "dRTauLep", "mbL", "mTauTauVis", "pTTauTauVis",
    "T1", "mTtau0", "HT", "dEtaHH", "dPhiTauTauMET", "dRB1Lep0",
    "tau_pt", "tau_eta", "tau_prong",   # prong is a primary FF axis; the GNN sees it via objects
]


def load_events(data_files):
    arrs = [h5py.File(f, "r")["events"][:] for f in data_files]
    return np.concatenate(arrs) if len(arrs) > 1 else arrs[0]


def find_prob_field(pred_file):
    """Locate the pID / pantiID columns in a SALT prediction file."""
    with h5py.File(pred_file, "r") as f:
        for key in f:
            ds = f[key]
            if ds.dtype.names is None:
                continue
            names = ds.dtype.names
            pid = next((n for n in names if n.endswith("pID")), None)
            pan = next((n for n in names if n.endswith("pantiID")), None)
            if pid and pan:
                return key, pid, pan
    raise RuntimeError("Could not find pID/pantiID columns in prediction file "
                       f"{pred_file}. Datasets: {list(h5py.File(pred_file,'r').keys())}")


def get_pID_real(pred_file, n_expected):
    key, pid, pan = find_prob_field(pred_file)
    with h5py.File(pred_file, "r") as f:
        arr = f[key][:]
    p = arr[pid].astype(float)
    # normalise defensively (softmax outputs already sum to 1)
    denom = arr[pid].astype(float) + arr[pan].astype(float)
    p = np.where(denom > 0, arr[pid].astype(float) / denom, 0.0)
    if len(p) != n_expected:
        raise RuntimeError(f"prediction rows ({len(p)}) != data rows ({n_expected}); "
                           "pass the same fakeCR file that was used for prediction.")
    return p


def get_pID_standin(ev):
    """Train a quick GBDT on DATA-only ID-vs-antiID to stand in for the GNN."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    is_data = ev["is_data"] == 1
    X = np.column_stack([ev[f][is_data] for f in STANDIN_FEATURES])
    y = ev["is_idfake"][is_data].astype(int)
    print(f"  stand-in: training GBDT on {is_data.sum():,} DATA events "
          f"(ID={int((y==1).sum()):,} / antiID={int((y==0).sum()):,}) ...")
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         validation_fraction=0.1, random_state=0)
    clf.fit(X, y)
    # predict p_ID for ALL events (needed for the anti-ID averaging below)
    Xall = np.column_stack([ev[f] for f in STANDIN_FEATURES])
    return clf.predict_proba(Xall)[:, 1]


def inclusive_scale(ev, p_ID):
    """c = (actual subtracted ID yield) / (yield the model predicts).

    Fitted on a fold the model neither trained on nor is applied to (i.e. the
    VALIDATION fold), this pins the one number the loss cannot constrain: the
    global FF normalisation. Fitting it on the apply fold would be selecting on
    the test set -- exactly what the 3-fold split exists to prevent.
    """
    p_ID = np.clip(p_ID, 1e-6, 1 - 1e-6)
    ff_x = p_ID / (1.0 - p_ID)
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"]
    pred = (w[idf == 0] * ff_x[idf == 0]).sum()
    actual = w[idf == 1].sum()
    return actual / pred, pred, actual


def closure(ev, p_ID, binned_weight, out_dir, tag, scale=1.0):
    os.makedirs(out_dir, exist_ok=True)
    p_ID = np.clip(p_ID, 1e-6, 1 - 1e-6)
    ff_x = scale * p_ID / (1.0 - p_ID)               # per-event GNN fake factor

    pt = np.clip(ev["tau_pt"], PT_EDGES[0], PT_EDGES[-1] - 1e-3)
    pbin = np.digitize(pt, PT_EDGES) - 1
    endc = ev["is_endcap"]; prong = ev["tau_prong"]
    idf = ev["is_idfake"]; isd = ev["is_data"]
    w = ev[binned_weight].astype(float)

    centers = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    lab = {0: "Barrel", 1: "Endcap"}
    tot_pred = tot_true = 0.0
    print(f"\n  {'category':16} {'ptbin':>9} {'binnedFF':>9} {'gnnFF':>8} {'ratio':>7}")
    for ax, (e, p) in zip(axes.ravel(), [(0, 1), (0, 3), (1, 1), (1, 3)]):
        binned, gnn = [], []
        for b in range(len(PT_EDGES) - 1):
            cat = (endc == e) & (prong == p) & (pbin == b)
            tl = w[cat & (idf == 1)].sum(); ta = w[cat & (idf == 0)].sum()
            binned.append(tl / ta if ta > 0 else np.nan)
            # The published FF is a ratio of SUBTRACTED yields, FF = A/B. Its exact
            # per-event analogue is the subtraction-WEIGHTED mean over the anti-ID
            # side: sum(w*FF(x)) / sum(w). That identity is what makes the two
            # comparable -- for a calibrated p(x) they agree bin by bin. Averaging
            # unweighted over anti-ID data instead silently drops the MC
            # subtraction (~6% overall, and not flat in x).
            m = cat & (idf == 0)
            denom = w[m].sum()
            pred = (w[m] * ff_x[m]).sum()
            gnn.append(pred / denom if denom != 0 else np.nan)
            tot_pred += pred          # predicted ID fake yield in this bin
            tot_true += tl            # actual subtracted ID yield
        binned = np.array(binned); gnn = np.array(gnn)
        for b in range(len(centers)):
            r = gnn[b] / binned[b] if (binned[b] and np.isfinite(binned[b])) else np.nan
            print(f"  {lab[e]+' '+str(p)+'-prong':16} {int(PT_EDGES[b]):>4}-{int(PT_EDGES[b+1]):>4} "
                  f"{binned[b]:>9.3f} {gnn[b]:>8.3f} {r:>7.2f}")
        ax.stairs(binned, PT_EDGES, color="k", lw=2, label="binned FF (data-trueMC)")
        ax.plot(centers, gnn, "o-", color="crimson", label="GNN  <p/(1-p)>")
        ax.set_title(f"{lab[e]}  {p}-prong")
        ax.set_xlabel(r"$\tau$ $p_T$ [GeV]"); ax.set_ylabel("fake factor")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # Aggregate closure: this is what actually propagates into the analysis.
    # Per-bin ratios can be biased in opposite directions and still cancel here --
    # or not. Both numbers matter, so print both.
    print(f"\n  {'INCLUSIVE closure':22} predicted {tot_pred:10,.1f}   actual {tot_true:10,.1f}"
          f"   ratio {tot_pred/tot_true:6.3f}" if tot_true else "")

    fig.suptitle(f"Fake-factor closure  ({tag})", fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, f"fakefactor_closure_{tag}.png")
    fig.savefig(out, dpi=130)
    print(f"\n  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="preprocessed fakeCR.h5 file(s)")
    ap.add_argument("--pred", default=None, help="SALT prediction h5 (real mode)")
    ap.add_argument("--standin-demo", action="store_true", help="local demo with a GBDT stand-in")
    ap.add_argument("--out-dir", default="fakefactor_closure_out")
    ap.add_argument("--calib-pred", default=None,
                    help="Prediction h5 on the CALIBRATION fold (use the validation "
                         "fold, never the apply fold). Fits one global FF scale.")
    ap.add_argument("--calib-data", default=None,
                    help="The fakeCR.h5 matching --calib-pred.")
    args = ap.parse_args()

    ev = load_events(args.data)
    print(f"loaded {len(ev):,} events from {len(args.data)} file(s)")

    if args.standin_demo:
        p_ID = get_pID_standin(ev)
        closure(ev, p_ID, binned_weight="is_data", out_dir=args.out_dir, tag="standin_raw")
    elif args.pred:
        p_ID = get_pID_real(args.pred, len(ev))
        scale, tag = 1.0, "salt"
        if args.calib_pred:
            if not args.calib_data:
                ap.error("--calib-pred requires --calib-data")
            cev = load_events([args.calib_data])
            cp = get_pID_real(args.calib_pred, len(cev))
            scale, pred, actual = inclusive_scale(cev, cp)
            print(f"calibration fold: predicted {pred:,.1f}  actual {actual:,.1f}"
                  f"  -> global FF scale = {scale:.4f}")
            print("  (one number, fitted off the apply fold; the SHAPE is untouched)")
            tag = "salt_calibrated"
        closure(ev, p_ID, binned_weight="fakefactor_weight", out_dir=args.out_dir,
                tag=tag, scale=scale)
    else:
        ap.error("provide either --pred <salt_prediction.h5> or --standin-demo")


if __name__ == "__main__":
    main()
