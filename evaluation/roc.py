import logging

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import scipy.interpolate as interp
from data_handler import DataHandler


def weighted_roc_curve(y_true, y_score, sample_weight, bin_edges=None, n_points=1000, safe_eps=1e-6):
    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    sample_weight = sample_weight[sorted_idx]
    is_sig = y_true == 1
    is_bkg = y_true == 0

    total_sig = np.sum(sample_weight[is_sig])
    total_bkg = np.sum(sample_weight[is_bkg])

    # Cumulative signal/background sums
    cum_sig = np.cumsum(sample_weight * is_sig)
    cum_bkg = np.cumsum(sample_weight * is_bkg)

    tpr_raw = np.concatenate(([0.0], cum_sig / (total_sig + safe_eps)))
    fpr_raw = np.concatenate(([0.0], cum_bkg / (total_bkg + safe_eps)))

    # Prepare cumulative background weight and weight^2
    w_bkg = sample_weight * is_bkg
    w_bkg2 = (sample_weight ** 2) * is_bkg
    cum_w_bkg = np.cumsum(w_bkg)
    cum_w2_bkg = np.cumsum(w_bkg2)

    # Insert zero at start to align with tpr/fpr
    cum_w_bkg = np.concatenate(([0.0], cum_w_bkg))
    cum_w2_bkg = np.concatenate(([0.0], cum_w2_bkg))

    # Compute effective n_bkg and σ_fpr vectorized
    n_eff = (cum_w_bkg ** 2) / (cum_w2_bkg + safe_eps)
    fpr_clipped = np.clip(fpr_raw, 0.0, 1.0)
    # sigma_fpr_raw = np.sqrt((fpr_clipped * (1 - fpr_clipped)) / (n_eff + safe_eps))
    # sigma_fpr_raw = sigma_fpr_raw / (fpr_clipped ** 2 + safe_eps)

    # Poisson-style uncertainty: σ = sqrt(sum w^2) / total_bkg
    sigma_fpr_raw = np.sqrt(cum_w2_bkg) / (total_bkg + safe_eps)
    # sigma_fpr_raw = sigma_fpr_raw / (fpr_clipped ** 2 + safe_eps)

    # Interpolate everything to fixed TPR grid
    tpr_uniform = np.linspace(0, 1, n_points)
    fpr_interp = np.interp(tpr_uniform, tpr_raw, fpr_clipped)
    sigma_fpr_interp = np.interp(tpr_uniform, tpr_raw, sigma_fpr_raw)

    return fpr_interp, tpr_uniform, sigma_fpr_interp


def draw_roc(
        ax, y_true, y_score, tag: str, sample_weight=None,
        bin_edges=None,
        ref_fpr=None, ref_tpr=None, ref_n_weight=None, ref_unc=None,
):
    """Compute ROC curve and AUC and plot on the given axis."""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    fpr, tpr, _ = weighted_roc_curve(y_true, y_score, sample_weight, bin_edges)
    auc = np.trapz(tpr, fpr)

    # Plot the ROC curve on the provided axis
    ax[0].plot(fpr, tpr, label=f"{tag} (AUC = {auc:.3f})", linewidth=2)
    return auc


def draw_roc_inverse(
        ax_list, y_true, y_score, tag: str, sample_weight=None,
        bin_edges=None,
        ref_fpr=None, ref_tpr=None, ref_n_weight=None, ref_unc=None,
        safe_eps=1e-6
):

    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    ax_upper, ax_lower = ax_list

    # Compute ROC for current data
    fpr, tpr, fpr_unc = weighted_roc_curve(y_true, y_score, sample_weight, bin_edges)

    # Compute AUC
    auc = np.trapz(tpr, fpr)

    fpr = fpr.astype(np.float64)
    tpr = tpr.astype(np.float64)
    sample_weight = sample_weight.astype(np.float64)

    # Compute background rejection (1/FPR)
    safe_fpr = np.where(fpr > safe_eps, fpr, np.nan)  # Ensure no zeros in fpr
    background_rej = np.where(np.isnan(safe_fpr), 1. / safe_eps, 1. / safe_fpr)  # Compute safely

    # Plot the ROC curve on the upper axis
    ax_upper.plot(tpr, background_rej, label=f"{tag} (AUC = {auc:.3f})", linewidth=2)

    # If reference ROC is provided, interpolate it safely
    if ref_fpr is not None and ref_tpr is not None and ref_n_weight is not None and ref_unc is not None:
        assert len(ref_fpr) == len(ref_tpr), "Reference FPR and TPR must have the same length"
        assert len(ref_fpr) == len(fpr), f"Reference FPR {len(ref_fpr)} must match current FPR {len(fpr)} length"

        ref_fpr = ref_fpr.astype(np.float64)
        ref_tpr = ref_tpr.astype(np.float64)
        ref_n_weight = ref_n_weight.astype(np.float64)

        # Clip reference FPR
        ref_fpr_clipped = np.clip(ref_fpr, safe_eps, None)
        background_rej_ref = 1. / ref_fpr_clipped

        # Compute ratio and its uncertainty
        ratio = background_rej / background_rej_ref
        sigma_ratio = ratio * np.sqrt(
            (fpr_unc / (fpr + safe_eps)) ** 2 +
            (ref_unc / (ref_fpr_clipped + safe_eps)) ** 2
        )

        # Plot the ratio on the lower axis with correct uncertainty
        ax_lower.plot(tpr, ratio, label=f"{tag} Ratio", linewidth=2)
        ax_lower.fill_between(tpr, ratio - sigma_ratio, ratio + sigma_ratio, alpha=0.2)

    return auc


# Define a custom sorting function (modify as needed)
def custom_sort(label, labels):
    def extract_group(label):
        match = re.search(r'\[(.*?)]', label)  # Extracts text inside brackets
        return match.group(1) if match else "zzz"  # Ensure "Random Classifier" is last

    # Sort by extracted group, keeping "Random Classifier" last
    sorted_labels = sorted(labels, key=lambda x: (extract_group(x), x if "Random" not in x else "zzz"))
    return sorted_labels.index(label)


def sort_legend(ax, title):
    """Sort legend labels and update the plot legend."""
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: custom_sort(x[1], labels))
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    ax.legend(sorted_handles, sorted_labels, title=title)


def process_data(data_handler, drop_negative_weights):
    """Process data and handle negative weights if required."""
    data = data_handler.combined_data
    empty = all(df.empty for data_df in data.values() for df in data_df.values())
    return {**data_handler.fold_data, 'combined': data} if not empty else data_handler.fold_data


def extract_data(data_df, score, data_handler, drop_negative_weights):
    """Extract y_true, y_score, and sample_weight, handling negative weights if needed."""
    y_true = data_df[data_handler.label_col].values
    y_score = data_df[score].values
    sample_weight = data_df[data_handler.weight_col].values
    if drop_negative_weights:
        mask = sample_weight > 0
        return y_true[mask], y_score[mask], sample_weight[mask]
    return y_true, y_score, sample_weight


def plot_per_fold(
        ax, fold, data, data_handler, drop_negative_weights,
        reference_dataset, reference_classifier,
        roc_results, draw_function, logger):
    """Plot ROC for each fold."""

    # first check if references are available
    if reference_dataset in data and reference_classifier in data[reference_dataset]:
        reference_data = data[reference_dataset][reference_classifier]
        r_true, r_score, r_w = extract_data(reference_data, reference_classifier, data_handler, drop_negative_weights)
        ref_fpr, ref_tpr, ref_unc = weighted_roc_curve(r_true, r_score, r_w, bin_edges=data_handler.default_bin_edges)
        ref_n_weight = r_w[r_true == 0]
    else:
        logger.error(f"Reference dataset/classifier not found: {reference_dataset}/{reference_classifier}")
        ref_fpr, ref_tpr, ref_n_weight, ref_unc = None, None, None, None

    for data_type, data_dict in data.items():
        if roc_results:
            roc_results[fold][data_type] = {}
        for score, config in data_handler.classifier_config.items():
            if score not in data_dict:
                continue
            data_df = data_dict[score]
            if data_df.empty:
                continue
            if not config['split_train_test_val'] and data_type != 'test':
                continue

            y_true, y_score, sample_weight = extract_data(data_df, score, data_handler, drop_negative_weights)

            auc = draw_function(
                ax, y_true, y_score, f"[{score}] {data_type.upper()} ", sample_weight,
                data_handler.default_bin_edges,
                ref_fpr, ref_tpr, ref_n_weight, ref_unc
            )

            if roc_results:
                roc_results[fold][data_type][score] = auc


def plot_per_dataset(
        ax, dataset_type, loop_data, data_handler, drop_negative_weights,
        reference_fold, reference_classifier,
        draw_function, logger
):
    """Plot ROC curves comparing different folds for a dataset type."""

    for fold, data_dict in loop_data.items():
        if fold == 'combined' or dataset_type not in data_dict:
            continue
        for score, config in data_handler.classifier_config.items():
            if score not in data_dict[dataset_type]:
                continue

            # first check if references are available
            if (
                    reference_fold in loop_data
                    and
                    reference_classifier in loop_data[reference_fold][dataset_type]
            ):
                reference_data = loop_data[reference_fold][dataset_type][reference_classifier]
                r_true, r_score, r_w = extract_data(
                    reference_data, reference_classifier, data_handler, drop_negative_weights
                )
                ref_fpr, ref_tpr, ref_unc = weighted_roc_curve(r_true, r_score, r_w,
                                                               bin_edges=data_handler.default_bin_edges)
                ref_n_weight = r_w[r_true == 0]
            else:
                logger.error(f"Reference dataset/classifier not found: {reference_fold}/{reference_classifier}")
                ref_fpr, ref_tpr, ref_n_weight, ref_unc = None, None, None, None

            data_df = data_dict[dataset_type][score]
            if data_df.empty:
                continue
            if not config['split_train_test_val'] and dataset_type != 'test':
                continue

            y_true, y_score, sample_weight = extract_data(data_df, score, data_handler, drop_negative_weights)
            draw_function(
                ax, y_true, y_score, f"[{score}] {fold} ", sample_weight,
                data_handler.default_bin_edges,
                ref_fpr, ref_tpr, ref_n_weight, ref_unc
            )


def save_figure(fig, path: Path, filename: str):
    """Helper function to save a figure as PNG and PDF."""
    path.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(path / f"{filename}.{ext}", dpi=300)


def roc(
        data_handler: DataHandler,
        roc_range,
        reference_dataset,
        reference_classifier,
        reference_fold,
        figsize=(8, 6),
        save_path=None,
        drop_negative_weights=False,
        logger=None,
        include_roc=True,
        include_roc_inverse=True,
):
    """Plot ROC for each fold and dataset type."""
    roc_results = {}
    loop_data = process_data(data_handler, drop_negative_weights)

    plot_type_config = []
    if include_roc:
        plot_type_config.append({
            'save_name': 'roc',
            'draw_function': draw_roc,
            'x_title': "False Positive Rate (FPR)",
            'y_title': "True Positive Rate (TPR)",
            'y_log': False,
            'x_range': None,
            'record_auc': True,
            'random_classifier': True,
            'ratio': False
        })
    if include_roc_inverse:
        plot_type_config.append({
            'save_name': 'roc_inverse',
            'draw_function': draw_roc_inverse,
            'x_title': "True Positive Rate (TPR)",
            'y_title': "Background Rejection (1/FPR)",
            'y_log': True,
            'x_range': roc_range,
            'record_auc': False,
            'random_classifier': False,
            'ratio': True,
            'n_bins': data_handler.base_bin_num,
        })

    logger.info("[roc] --> Plotting ROC for each fold")
    for fold, data in loop_data.items():
        if all(df.empty for data_df in data.values() for df in data_df.values()):
            continue
        roc_results[fold] = {}

        for config in plot_type_config:
            if config['ratio']:
                fig, ax_list = plt.subplots(
                    2, 1,
                    figsize=figsize,
                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
                    sharex=True,
                )
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax_list = [ax]

            plot_per_fold(
                ax_list, fold, data, data_handler, drop_negative_weights,
                reference_dataset, reference_classifier,
                roc_results if config['record_auc'] else None,
                draw_function=config['draw_function'],
                logger=logger,
            )
            if config['random_classifier']:
                ax_list[0].plot([0, 1], [0, 1], 'k--', label="Random Classifier")

            ax_list[-1].set_xlabel(config['x_title'])
            ax_list[0].set_ylabel(config['y_title'])
            sort_legend(ax_list[0], f"ROC: {fold}")

            for ax in ax_list:
                ax.grid()

            if config['y_log']:
                ax_list[0].set_yscale('log')

            if config['x_range']:
                ax_list[-1].set_xlim(config['x_range'][0], config['x_range'][1])

            if save_path:
                save_figure(fig, save_path / fold, config["save_name"])
            plt.close(fig)

    logger.info("[roc] --> Plotting ROC for each dataset")
    for dataset_type in data_handler.dataset_types:
        if all(
                score not in loop_data[fold][dataset_type] or loop_data[fold][dataset_type][score].empty
                for fold in data_handler.get_fold_names()
                for score in data_handler.classifier_config.keys()
        ):
            continue

        for config in plot_type_config:
            if config['ratio']:
                fig, ax_list = plt.subplots(
                    2, 1,
                    figsize=figsize,
                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
                    sharex=True,
                )
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax_list = [ax]

            plot_per_dataset(
                ax_list, dataset_type, loop_data, data_handler, drop_negative_weights,
                reference_fold, reference_classifier,
                draw_function=config['draw_function'],
                logger=logger,
            )
            if config['random_classifier']:
                ax_list[0].plot([0, 1], [0, 1], 'k--', label="Random Classifier")

            ax_list[-1].set_xlabel(config['x_title'])
            ax_list[0].set_ylabel(config['y_title'])
            sort_legend(ax_list[0], f"ROC: {dataset_type}")

            for ax in ax_list:
                ax.grid()

            if config['y_log']:
                ax_list[0].set_yscale('log')

            if config['x_range']:
                ax_list[-1].set_xlim(config['x_range'][0], config['x_range'][1])

            if save_path:
                save_figure(fig, save_path / 'fold_comparison', f"{config['save_name']}_{dataset_type}")

            plt.close(fig)

    return roc_results
