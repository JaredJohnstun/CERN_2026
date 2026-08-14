#!/usr/bin/env python3
"""
Compare calibrated GNN fake factors from the three run_fold.sh runs

Each model is evaluated on its own unseen application fold, using the same
one-number calibration and subtraction-weighted anti-ID averaging as
fakefactor_closure.py. This measures fold-to-fold stability; it is not a
same-event comparison because the three application folds differ.
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT_EDGES = np.array([20, 30, 40, 50, 60, 80, 100, 400], float)
#Each tuple represents (is_endcapt, tau_prong) 0-> Barrel, 1-> Endcap
CATEGORIES = [(0, 1), (0, 3), (1, 1), (1, 3)]
REGION = {0: "Barrel", 1: "Endcap"}

def load_events(path):
    with h5py.File(path, "r") as f:
        return f["events"][:]

#Reads the GNN's predicted probabilities from a SALT Prediction File
def get_pid(path, n_expected):
    with h5py.File(path, "r") as f:
        #Search Every top-level dataset
        for key in f:
            ds = f[key]
            #Check whether it has named columns
            if ds.dtype.names is None:
                continue
            #Search for columns ending in pID and pantiID (prediction ID and prediction antiID)
            pid_name = next((n for n in ds.dtype.names if n.endswith("pID")), None)
            pan_name = next((n for n in ds.dtype.names if n.endswith("pantiID")), None)
            if pid_name and pan_name:
                arr = ds[:]
                pid = arr[pid_name].astype(float)
                pan = arr[pan_name].astype(float)
                #Make sure they're normalized
                den = pid + pan
                p = np.where(den > 0, pid / den, 0.0)
                #Ensure prediction array contains exactly one prediction for every event in corresponding file
                if len(p) != n_expected:
                    raise RuntimeError(
                        f"{path}: {len(p):,} prediction rows, expected {n_expected:,}"
                    )
                return p
    #Raise Error if zero ID's are found in file
    raise RuntimeError(f"{path}: no pID/pantiID fields found")

#Convert GNN probability into a fake factor
#Scale is a calibration constant
def ff_from_p(p, scale=1.0):
    #recall FF(GNN) = pID(x) / (1 - pID(x))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return scale * p / (1 - p)

#Calculates the global calibration constant for a fold model
#Adjusts overall normalization but does not change the relative shape
def calibration_scale(ev, p):
    #convert probabilities into uncalibrated fake factors
    ff = ff_from_p(p)
    #retrieve event weights and ID classifications
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"] #1 -> ID fake region, 0 -> anti_ID fake region
    #Select antiID events, muiltiply each event weight by its FF, and sum all results (prediction)
    pred = (w[idf == 0] * ff[idf == 0]).sum()
    #Actual yield is summ of each event weight in ID region
    actual = w[idf == 1].sum()
    #Calibration constant is actual / pred
    #Returns Scale, uncalibrated predicted yield, and the actual yield
    return (actual / pred if pred != 0.0 else 1.0), pred, actual

#Central comparison function
def per_bin(ev, ff):
    #Force every event into the supported range
    pt = np.clip(ev["tau_pt"], PT_EDGES[0], PT_EDGES[-1] - 1e-3)
    #Assign a bin number to every event. The minus one converts numbering into zero-based indexing
    pbin = np.digitize(pt, PT_EDGES) - 1
    #Read necessary event properties. They have one value per event
    endc = ev["is_endcap"].astype(int)
    prong = ev["tau_prong"].astype(int)
    idf = ev["is_idfake"]
    w = ev["fakefactor_weight"].astype(float)
    out = {} #output dictionary that will be returned

    #Loop through our four categories
    for e, p in CATEGORIES:
        vals = []
        #for each category, go through each bin (8 bin edges means 7 bins)
        for b in range(len(PT_EDGES) - 1):
            #Check that the event we're on is the one we want
            #Grabs all events at the same time that satisfy these 4 conditions:
            #Is/is not endcap/barrel
            #Has proper prongness
            #Is in correct pt bin
            #is in anti_ID fake region
            m = (endc == e) & (prong == p) & (pbin == b) & (idf == 0) #m used for "Boolean mask"
            #Denominator is the sum of the weights of every selected event (Event weight)
            den = w[m].sum()
            #Numerator is the weighted sum of the Fake Factors
            num = (w[m] * ff[m]).sum()
            #Append weighted average of the Fake Factors to the array
            vals.append(num / den if den != 0 else np.nan)
        #Save results by event category
        out[(e, p)] = np.asarray(vals)
    return out

#Performs the complete calculation for one model fold
def process(label, pred, data, calib_pred, calib_data):
    #load application data (function calls from above)
    #Loads application events and GNN probabilites
    ev = load_events(data)
    p = get_pid(pred, len(ev))
    #Load calibration-fold events and GNN probabilites
    cev = load_events(calib_data)
    cp = get_pid(calib_pred, len(cev))
    #Determine the global scale factor using only the calibration fold
    scale, calib_prediction, calib_actual = calibration_scale(cev, cp)
    #Apply the Scale and turn the application-fold probabilities into calibrated event-level FF's
    ff = ff_from_p(p, scale)
    #Calculate binned averges for each of the 4 categories
    values = per_bin(ev, ff)

    #After calibrating the GNN, does it correctly predict the # of ID fake events?
    #start by grabbing event weights
    w = ev["fakefactor_weight"].astype(float)
    idf = ev["is_idfake"]
    #Generate Prediction of # of fake events
    #Each Event contributes its weight times its FF, and we'll sum all events together
    #This is predicted number of fake ID events using anti_ID events
    apply_prediction = (w[idf == 0] * ff[idf == 0]).sum()
    #Now look at actual fake ID events
    apply_actual = w[idf == 1].sum()
    #Calculate closure
    closure = apply_prediction / apply_actual if apply_actual != 0 else np.nan
    #Print some valuable info to the screen
    print(
        f"{label}: scale={scale:.6f} | calibration {calib_prediction:,.1f}/{calib_actual:,.1f} "
        f"| apply closure={closure:.4f}"
    )
    #return the fold name, FF curve values, and closure value for plotting later
    return {"label": label, "values": values, "closure": closure}

#Prints table to the screen comparing folds
def print_table(folds):
    print("\nCalibrated GNN fake factor by category and tau-pT bin")
    #python jargon to control formatting
    header = f"{'category':<18}{'pT bin':>12}" + "".join(
        f"{f['label']:>14}" for f in folds
    ) + f"{'std dev':>14}"
    print(header)
    print("-" * len(header))

    for cat in CATEGORIES:
        e, p = cat
        name = f"{REGION[e]} {p}-prong"
        for b in range(len(PT_EDGES) - 1):
            #Gets the proper value sorting from the proper bin from the proper curve
            vals = np.array([f["values"][cat][b] for f in folds], float)
            #Remove invalid values
            finite = vals[np.isfinite(vals)]
            #calculate sample standard deviation of the three fold values
            #ddof = 1 means sample standard deviation convention
            spread = np.std(finite, ddof=1) if len(finite) > 1 else np.nan
            #more python formatting jargon to print the pretty table
            row = f"{name:<18}{int(PT_EDGES[b]):>5}-{int(PT_EDGES[b+1]):<6}"
            row += "".join(
                f"{v:>14.5f}" if np.isfinite(v) else f"{'nan':>14}" for v in vals
            )
            row += f"{spread:>14.5f}" if np.isfinite(spread) else f"{'nan':>14}"
            print(row)

#Creates .png output
def make_plot(folds, out_dir):
    #Create output directory
    os.makedirs(out_dir, exist_ok=True)
    #Calculate bin centers
    centers = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])
    #Create 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    #Pair panels with categories
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        e, p = cat
        #Draw three model-fold curves
        for fold in folds:
            ax.plot(centers, fold["values"][cat], marker="o", linewidth=1.8,
                    label=fold["label"])
            ax.set_title(f"{REGION[e]} {p}-prong")
            ax.set_xlabel(r"$\tau$ $p_T$ [GeV]")
            ax.set_ylabel("Calibrated GNN fake factor")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("GNN fake-factor comparison across model folds", fontsize=14)
    fig.tight_layout()
    #Save and close
    out = os.path.join(out_dir, "gnn_fakefactor_fold_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

#Function to automatically find the prediction files
def find_single(pattern, description):
    import glob #Lets python search using wildcard patterns
    #Returns every matching specified pattern
    matches = sorted(glob.glob(pattern))
    #Case if no files found
    if not matches:
        raise RuntimeError(f"could not find {description}: {pattern}")
    #Case if several files are found
    if len(matches) > 1:
        joined = "\n    ".join(matches)
        raise RuntimeError(
            f"found more than one {description}; keep only the desired file:\n    {joined}"
        )
    return matches[0]

def main():
    #Defines command line interface
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    #Specifies where to find fold directories.
    ap.add_argument(
        "--input-dir",
        default="gnn_fold_inputs",
        help="Directory containing fold0/, fold1/, and fold2/ prediction files",
    )
    #Specification for where to get event level information to interpret prediction files
    ap.add_argument(
        "--preprocessed",
        default=os.environ.get(
            "PP", "/home/shahzad/ML_Fakes_LepHad/inputData/preprocessed"
        ),
        help="Root containing fold_0/fakeCR.h5, fold_1/fakeCR.h5, fold_2/fakeCR.h5",
    )
    #Argument for what you want the output directory to be
    ap.add_argument("--out-dir", default="gnn_fold_comparison_out")
    args = ap.parse_args()
    
    #Fold rotation dictionaries
    calibration_fold = {0: 1, 1: 2, 2: 0}
    application_fold = {0: 2, 1: 0, 2: 1}

    folds = []
    #Each iteration of loop handles one trained model
    for k in range(3):
        #Finds the fold directory
        fold_dir = os.path.join(args.input_dir, f"fold{k}")
        
        #Finds the application prediction
        pred_path = find_single(
            os.path.join(fold_dir, "*__test_fakeCR.h5"),
            f"fold {k} application prediction",
        )

        #Finds the calibration prediction
        calib_pred_path = find_single(
            os.path.join(
                fold_dir,
                f"*__calib_fold{calibration_fold[k]}.h5",
            ),
            f"fold {k} calibration prediction",
        )
        
        #Builds the application data path
        data_path = os.path.join(
            args.preprocessed,
            f"fold_{application_fold[k]}",
            "fakeCR.h5",
        )

        #Builds the calibration data path
        calib_data_path = os.path.join(
            args.preprocessed,
            f"fold_{calibration_fold[k]}",
            "fakeCR.h5",
        )
        
        #Process and store the model
        folds.append(
            process(
                f"model fold {k}",
                pred_path,
                data_path,
                calib_pred_path,
                calib_data_path,
            )
        )
    #Output computations
    print_table(folds)
    make_plot(folds, args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
