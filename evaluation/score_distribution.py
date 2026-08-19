import logging

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from pathlib import Path
from scipy.stats import chisquare, wasserstein_distance

from data_handler import DataHandler
from transform_binning import get_rebin_bins, convert_indices_to_edges, convert_indices_to_edges_under_overflow, \
    calculate_binned_significance, root_to_numpy_histogram


def rebin_histogram(df, new_bin_edges, old_bin_edges):
    """Rebins a histogram DataFrame based on a numpy array of new bin edges."""

    # Ensure new_bin_edges and old_bin_edges are numpy arrays
    new_bin_edges = np.asarray(new_bin_edges)
    old_bin_edges = np.asarray(old_bin_edges)
    old_bin_centers = 0.5 * (old_bin_edges[:-1] + old_bin_edges[1:])

    # Define histogram and uncertainty columns
    hist_columns = [col for col in df.columns if "_unc" not in col]  # Histogram values
    unc_columns = [col for col in df.columns if "_unc" in col]  # Uncertainty values

    # Initialize rebinned histograms and uncertainties
    rebinned_hist = np.zeros((len(new_bin_edges) - 1, len(hist_columns)))
    rebinned_unc = np.zeros((len(new_bin_edges) - 1, len(unc_columns)))

    # Loop over new bins and sum up corresponding old bins
    for i in range(len(new_bin_edges) - 1):
        # Find indices where old bin centers fall inside the new bin
        mask = (old_bin_centers >= new_bin_edges[i]) & (old_bin_centers <= new_bin_edges[i + 1])

        # Sum the histogram values in this bin
        rebinned_hist[i] = df[hist_columns].values[mask].sum(axis=0)

        # Propagate uncertainties (sqrt of sum of squares)
        rebinned_unc[i] = np.sqrt((df[unc_columns].values[mask] ** 2).sum(axis=0))

    # Create new DataFrame with rebinned values
    rebinned_df = pd.DataFrame(np.hstack([rebinned_hist, rebinned_unc]), columns=hist_columns + unc_columns)

    return rebinned_df


def empty_negative_bins(df: pd.DataFrame, data_handler: DataHandler) -> pd.DataFrame:
    # flip to get sample per label
    label_to_sample = {}
    for sample, label in data_handler.sample_to_label.items():
        label_to_sample.setdefault(label, []).append(sample)

    # drop signal and background histograms
    for label, sample_list in label_to_sample.items():
        df.drop(columns=[f"{label}", f"{label}_unc"], inplace=True)
        df[f"{label}"] = 0.0
        df[f"{label}_unc"] = 0.0

        # zero-clip per sample
        for samp in sample_list:
            # some samples may be empty after selection
            if samp not in df.columns:
                continue
            neg = df[samp] < 0.0
            df.loc[neg, samp] = 0.0
            # leave uncertainty

            # now sum per sample
            df[f"{label}"]     += df[samp]
            df[f"{label}_unc"] = (df[f"{label}_unc"]**2 + df[f"{samp}_unc"]**2).pow(0.5)

    return df


def draw_mva_distribution(
        ax,
        hist_df: pd.DataFrame,
        bin_edges: np.array,
        category_map,
        draw_config,
        name,
        density: bool = True,
        reference_dataset_hist: pd.DataFrame = None
):
    # Define colors for categories

    # Loop over categories and plot histograms
    for label, category in category_map.items():

        label_name = f"[{name:^10}] {category.get('name', label)}"
        # if reference_dataset_hist is None:
        #     label_name = f"[{name:^10}] {category.get('name', label)}: Reference dataset"
        # else:
        #     cur_data = hist_df[str(label)]
        #     ref_data = reference_dataset_hist[str(label)]
        #
        #     f_obs = np.clip(cur_data, 1e-6, None)
        #     f_exp = np.clip(ref_data, 1e-6, None)
        #
        #     chi_stat, _ = chisquare(
        #         f_obs=f_obs,
        #         f_exp=f_exp * (np.sum(f_obs) / np.sum(f_exp))  # normalize to same area
        #     )
        #
        #     wd = wasserstein_distance(
        #         cur_data, ref_data,
        #         u_weights=np.ones_like(cur_data) / len(cur_data),
        #         v_weights=np.ones_like(ref_data) / len(ref_data)
        #     )
        #
        #     label_name = (
        #         f"[{name:^10}] {category.get('name', label)}: "
        #         f"$\\chi^2 = {chi_stat:.2f}$, $EMD = {wd:.2f}$"
        #     )

        weights = hist_df[str(label)].values
        scores = (bin_edges[:-1] + bin_edges[1:]) / 2

        draw_config_ = draw_config.copy()
        draw_type = draw_config_.pop("type", "hist")

        if draw_type == "hist":
            ax.hist(
                scores, bins=bin_edges, weights=weights, density=density, alpha=0.5,
                label=label_name,
                color=category.get("color", "gray"),
                **draw_config_
            )
        elif draw_type == "marker":
            bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist, _ = np.histogram(scores, bins=bin_edges, weights=weights, density=density)

            ax.plot(
                bin_center, hist, label=label_name, color=category.get("color", "gray"),
                **draw_config_
            )
        else:
            raise ValueError(f"Unknown draw type {draw_type}")


def draw_ratio_plot(
        ax, hist_df: pd.DataFrame, bin_edges: np.array, category_map, draw_config, name, reference_df,
        ratio_pulls: bool = False
):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers

    for category in category_map.keys():
        # Extract reference histogram (denominator)
        ref_hist = reference_df[f"{category}"].values
        ref_unc = reference_df[f"{category}_unc"].values

        # Extract current histogram (numerator)
        num_hist = hist_df[f"{category}"].values
        num_unc = hist_df[f"{category}_unc"].values

        # Compute ratio and propagate uncertainty
        ratio = np.divide(num_hist, ref_hist, out=np.ones_like(num_hist), where=ref_hist != 0)
        ratio_unc = np.abs(ratio) * np.sqrt(
            np.divide(num_unc, num_hist, out=np.zeros_like(num_unc), where=num_hist != 0) ** 2 +
            np.divide(ref_unc, ref_hist, out=np.zeros_like(ref_unc), where=ref_hist != 0) ** 2
        )

        draw_config_ = copy.deepcopy(draw_config)
        if 'type' in draw_config_:
            draw_config_.pop("type")
        if 'histtype' in draw_config_:
            draw_config_.pop("histtype")

        # Plot ratio with error bars
        if not ratio_pulls:
            ax.errorbar(
                bin_centers,
                ratio,
                yerr=ratio_unc,
                fmt=draw_config_.pop("marker").get(category, "o"),
                label=f"{name}",
                color=draw_config_["ratio_color"].get(category, "gray"),
                markersize=4,
                capsize=4,
            )
        else:
            ratio = np.divide(
                num_hist - ref_hist, np.sqrt(num_unc ** 2 + ref_unc ** 2),
                out=np.zeros_like(num_hist), where=(num_unc ** 2 + ref_unc ** 2) != 0
            )

        ax.plot(
            bin_centers, ratio,
            alpha=0.7,
            color=draw_config_.pop("ratio_color").get(category, "gray"),
            linestyle=draw_config_.pop("linestyle").get(category, "-"),
            **draw_config_
        )


def scores(
        data_handler: DataHandler,
        category: dict[int, dict[str, str]],
        reference_dataset: str,
        baseline_classifier: str,
        x_min: float = -1,
        x_max: float = 1,
        trafo: dict = None,
        bin_num: int = 200,
        figsize: tuple = (8, 6),
        y_log: bool = False,
        ratio_pulls: bool = False,
        fold_colors: dict[str, str] = None,
        fold_reference: str = None,
        save_path: Path = None,
        logger: logging.Logger = None,
        use_test_bin_edges: bool = False,
):
    # first, we need to get the scores from the data_handler and make histograms
    sig_results = {}

    h_sig = next((k for k, v in category.items() if v["name"] == "Signal"), None)
    h_bkg = next((k for k, v in category.items() if v["name"] == "Background"), None)

    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max")
    if not isinstance(x_min, (int, float)) or not isinstance(x_max, (int, float)):
        raise ValueError("x_min and x_max must be integers or floats")

    trafo_use_ROOT = True
    if trafo is not None:
        try:
            import ROOT as r

            file_dir = Path(__file__).parent.parent

            r.gInterpreter.AddIncludePath(f"{file_dir}/TransformTool")
            r.gInterpreter.ProcessLine(f".L {file_dir}/TransformTool/Root/HistoTransform.cxx+")
            r.gInterpreter.ProcessLine(f".include {file_dir}/TransformTool/HistoTransform.h")

        except ImportError:
            logger.warn("ROOT is required for transformation, but not found. Using self-made python instead.")
            trafo_use_ROOT = False

    loop_data = copy.deepcopy(data_handler.hists)

    #################################
    #            REBIN              #
    #   iterate loop_data to rebin  #
    #################################
    logger.info("[Score Distribution] --> Rebinning histograms")
    for classifier, score_data in loop_data.items():
        for fold, data in score_data.items():
            # loop start from test dataset
            data_types = list(data.keys())
            if 'test' in data_types: 
                data_types.remove('test')
                data_types = ['test'] + data_types
            for data_type in data_types:
                data_hist = data[data_type]
                if data_hist['hist'].empty:
                    continue

                # rebin histograms
                rebin_trafo_bins = None
                if trafo is None:
                    rebin_bin_edges = np.linspace(x_min, x_max, bin_num + 1)
                    rebin_hist = rebin_histogram(
                        data_hist['hist'], old_bin_edges=data_hist['bin_edges'], new_bin_edges=rebin_bin_edges
                    )
                else:
                    if trafo_use_ROOT:
                        bins = data_hist['bin_edges']
                        hist = data_hist['hist']

                        suffix = f"{classifier}_{fold}_{data_type}"
                        h_sig_r = r.TH1D(f"h_sig_{suffix}", f"h_sig_{suffix}", len(bins) - 1, bins)
                        for i in range(len(bins) - 1):
                            h_sig_r.SetBinContent(i + 1, hist[str(h_sig)].values[i])
                            h_sig_r.SetBinError(i + 1, hist[f"{h_sig}_unc"].values[i])

                        h_bkg_r = r.TH1D(f"h_bkg_{suffix}", f"h_bkg_{suffix}", len(bins) - 1, bins)
                        for i in range(len(bins) - 1):
                            h_bkg_r.SetBinContent(i + 1, hist[str(h_bkg)].values[i])
                            h_bkg_r.SetBinError(i + 1, hist[f"{h_bkg}_unc"].values[i])

                        method = 60
                        # maxUnc = trafo['max_unc']
                        histoTrafo = r.HistoTransform()
                        histoTrafo.trafoSixY = trafo['trafo_six_y']
                        histoTrafo.trafoSixZ = trafo['trafo_six_z']
                        histoTrafo.trafoSixtyMCLowBound = trafo['trafo_sixty_mc_low_bound']
                        histoTrafo.trafoSixMCstatUpBound = trafo['trafo_six_mc_stat_up_bound']
                        histoTrafo.trafoSixtyIncludeS = False

                        # find bins form histogram
                        if use_test_bin_edges and data_type != 'test':
                            # use test bin edges
                            rebin_trafo_bins = loop_data[classifier][fold]['test']['raw_trafo_edges']
                        else:
                            rebin_trafo_bins = histoTrafo.getRebinBins(h_bkg_r, h_sig_r, method)
                        # rebin the histograms
                        histoTrafo.rebinHisto(h_bkg_r, rebin_trafo_bins, True, False)
                        histoTrafo.rebinHisto(h_sig_r, rebin_trafo_bins, True, False)

                        h_sig_r = root_to_numpy_histogram(h_sig_r)
                        h_bkg_r = root_to_numpy_histogram(h_bkg_r)

                        rebin_bin_edges = convert_indices_to_edges_under_overflow(rebin_trafo_bins, bins)[::-1]
                    else:
                        # raise error of deprecated
                        raise NotImplementedError

                    rebin_hist = rebin_histogram(
                        data_hist['hist'], old_bin_edges=data_hist['bin_edges'], new_bin_edges=rebin_bin_edges
                    )


                # after rebinning the histograms, sample-per-sample we set negative bin values to zero
                if data_handler.has_sample_name:
                    rebin_hist = empty_negative_bins(rebin_hist, data_handler)


                loop_data[classifier][fold][data_type]['hist'] = rebin_hist
                loop_data[classifier][fold][data_type]['bin_edges'] = np.linspace(
                    data_hist['bin_edges'].min(), data_hist['bin_edges'].max(), len(rebin_hist) + 1
                )
                loop_data[classifier][fold][data_type]['rebin_edges'] = rebin_bin_edges
                loop_data[classifier][fold][data_type]['raw_trafo_edges'] = rebin_trafo_bins

    ##################################################
    #                PLOTTING                        #
    #   separate plots for each classifier per fold  #
    ##################################################
    logger.info("[Score Distribution] --> Plotting scores per classifier per fold")
    for classifier, score_data in loop_data.items():
        sig_results[classifier] = {}

        draw_config = data_handler.classifier_config[classifier]['score_plot_config']
        for fold, data in score_data.items():
            sig_results[classifier][fold] = {}
            empty_fold = sum([data_type['hist'].empty for data_type in data.values()]) == len(data)
            if empty_fold:
                continue

            # original distribution
            fig, (ax_up, ax_down) = plt.subplots(
                2, 1,
                figsize=figsize,
                # sharex=False,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
            )

            for data_type, data_hist in data.items():
                name = f"{data_type}"

                if data_hist['hist'].empty:
                    continue

                if data_type in draw_config:
                    draw_mva_distribution(
                        ax_up,
                        hist_df=data_hist['hist'],
                        bin_edges=data_hist['bin_edges'],
                        category_map=category,
                        draw_config=draw_config[data_type],
                        name=name, density=True,
                        reference_dataset_hist=loop_data[classifier][fold][reference_dataset][
                            'hist'] if data_type != reference_dataset else None,
                    )

                    split = data_handler.classifier_config[classifier]['split_train_test_val']
                    if data_type != reference_dataset and split:

                        # ratio plot
                        draw_ratio = True
                        if len(data_hist['bin_edges']) != len(
                                loop_data[classifier][fold][reference_dataset]['bin_edges']
                        ):
                            logger.error(
                                f"[{fold}] - [{classifier}]: Bin edges of {data_type} and {reference_dataset} do not match. "
                                f"Skipping ratio plot."
                            )
                            draw_ratio = False

                        ratio_draw_config = copy.deepcopy(draw_config[data_type])

                        ratio_draw_config['ratio_color'] = {
                            cat: cat_config.get('color', 'grey') for cat, cat_config in category.items()
                        }

                        ratio_draw_config['linestyle'] = {
                            cat: draw_config[data_type].get('linestyle', '-') for cat in category
                        }

                        ratio_draw_config['marker'] = {
                            cat: draw_config[data_type].get('marker', 'o') for cat in category
                        }

                        if draw_ratio:
                            draw_ratio_plot(
                                ax_down,
                                hist_df=data_hist['hist'],
                                bin_edges=data_hist['bin_edges'],
                                category_map=category,
                                draw_config=ratio_draw_config,
                                name=name,
                                reference_df=data[reference_dataset]['hist'],
                                ratio_pulls=ratio_pulls,
                            )

                    if trafo is not None:
                        for ax in [ax_up, ax_down]:
                            ax.set_xticks(
                                np.linspace(
                                    data_hist['bin_edges'].min(), data_hist['bin_edges'].max(),
                                    len(data_hist['bin_edges']) - 1
                                )
                            )
                            ax.set_xticklabels([i for i in range(len(data_hist['bin_edges']) - 1)])

                if h_sig is None or h_bkg is None:
                    raise ValueError("Signal or Background category not found in data")

                sig_ratio = data_hist['yields'].loc[str(h_sig)] / np.sum(data_hist['hist'][str(h_sig)].values)
                bkg_ratio = data_hist['yields'].loc[str(h_bkg)] / np.sum(data_hist['hist'][str(h_bkg)].values)

                # sig_results[classifier][fold][data_type] = np.sqrt(np.sum(calculate_binned_significance(
                #     h_s=data_hist['hist'][str(h_sig)].values, h_b=data_hist['hist'][str(h_bkg)].values,
                #     sig_scale=sig_ratio, bkg_scale=bkg_ratio,
                # ) ** 2))

                sig_results[classifier][fold][data_type] = calculate_binned_significance(
                    h_s=data_hist['hist'][str(h_sig)].values, h_b=data_hist['hist'][str(h_bkg)].values,
                    h_b_unc=data_hist['hist'][f"{h_bkg}_unc"].values,
                    sig_scale=sig_ratio, bkg_scale=bkg_ratio,
                )
                sig_results[classifier][fold][f'{data_type}_bin_edges'] = data_hist['rebin_edges'].tolist()
                sig_results[classifier][fold][f'{data_type}_signal_hists'] = data_hist['hist'][str(h_sig)].to_list()
                sig_results[classifier][fold][f'{data_type}_background_hists'] = data_hist['hist'][str(h_bkg)].to_list()
                sig_results[classifier][fold][f'{data_type}_background_unc_hists'] = data_hist['hist'][
                    f"{h_bkg}_unc"].to_list()
                if trafo is not None:

                    sig_results[classifier][fold][f'{data_type}_raw_trafo_bin_edges'] = list(data_hist['raw_trafo_edges'])

            has_data = any(ax_down.lines) or any(ax_down.collections) or any(ax_down.patches)

            handles, labels = ax_up.get_legend_handles_labels()
            sorted_handles_labels = sorted(
                zip(handles, labels), key=lambda hl: category[list(category)[0]]['name'] in hl[1]
            )
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

            ax_up.set_ylabel("A.U.")
            ax_up.legend(sorted_handles, sorted_labels)

            if y_log:
                ax_up.set_yscale('log')

            # Formatting ratio plot
            x_label = classifier if trafo is None else "Transformed " + classifier

            ax_down.axhline(1, linestyle="--", color="gray")  # Reference line at y=1
            ax_down.set_xlabel(x_label)
            if ratio_pulls:
                ax_down.set_ylabel(r"$Pulls [\sigma]$")
            else:
                ax_down.set_ylabel(f"Ratio to {reference_dataset}")
            # ax_down.legend()
            # ax_down.grid(True)

            if not has_data:
                fig.delaxes(ax_down)
                # fig.subplots_adjust(hspace=0.2)  # Adjust spacing after removing subplot
                fig.set_size_inches(figsize[0], figsize[1])

                ax_up.set_xlabel(x_label)

            if save_path is not None:
                fold_fig_path = save_path / fold
                # make directory if not exists
                fold_fig_path.mkdir(parents=True, exist_ok=True)
                # save png
                if trafo is None:
                    fold_fig_path = fold_fig_path / f"{classifier.replace(' ', '_')}_{fold}.png"
                else:
                    fold_fig_path = fold_fig_path / f"{classifier.replace(' ', '_')}_{fold}_trafo.png"
                fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')
                # save pdf
                fold_fig_path = fold_fig_path.with_suffix('.pdf')
                fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')

            # plt.show()
            plt.close(fig)

    #############################################################
    #                PLOTTING                                   #
    #   separate plots for each classifier for comparing folds  #
    #############################################################
    logger.info("[Score Distribution] --> Plotting scores per classifier for comparing folds")
    for classifier, score_data in loop_data.items():

        dataset_types = data_handler.dataset_types

        for dataset_type in dataset_types:

            all_empty = all([
                loop_data[classifier][fold][dataset_type]['hist'].empty
                for fold in data_handler.get_fold_names()
            ])

            if all_empty or fold_reference is None:  # skip if all classifiers are empty
                continue

            if loop_data[baseline_classifier][fold_reference][dataset_type]['hist'].empty:
                logger.warning(
                    f"[{classifier}] - [{dataset_type}]: Reference fold ({fold_reference}) is empty. "
                    f"Skipping fold comparison plot."
                )
                continue

            # plot scores comparison for each fold with all classifiers
            fig, (ax_main, ax_ratio,) = plt.subplots(
                2, 1,
                figsize=figsize,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
                sharex=True,
            )

            for fold in data_handler.get_fold_names():
                if fold == 'combined':
                    continue

                data = loop_data[classifier][fold][dataset_type]

                if data['hist'].empty:
                    continue

                for i, cat in enumerate(category.keys()):

                    name = f"{fold} - {category[cat]['name']}"
                    # if fold == fold_reference:
                    #     name = f"[{fold} - {category[cat]['name']}] Reference fold"
                    # else:
                    #     cur_data = data['hist'][str(cat)]
                    #     ref_data = loop_data[classifier][fold_reference][dataset_type]['hist'][str(cat)]
                    #
                    #     f_obs = np.clip(cur_data, 1e-6, None)
                    #     f_exp = np.clip(ref_data, 1e-6, None)
                    #
                    #     chi_stat, _ = chisquare(
                    #         f_obs=f_obs,
                    #         f_exp=f_exp * (np.sum(f_obs) / np.sum(f_exp))  # normalize to same area
                    #     )
                    #
                    #     wd = wasserstein_distance(
                    #         cur_data, ref_data,
                    #         u_weights=np.ones_like(cur_data) / len(cur_data),
                    #         v_weights=np.ones_like(ref_data) / len(ref_data)
                    #     )
                    #
                    #     name = (f"[{fold} - {category[cat]['name']}]: "
                    #             f"$\\chi^2 = {chi_stat:.2f}$, $EMD = {wd:.2f}$")

                    ax_main.hist(
                        x=(data['bin_edges'][:-1] + data['bin_edges'][1:]) / 2,
                        bins=data['bin_edges'],
                        weights=data['hist'][str(cat)],
                        density=True,
                        alpha=1.0,
                        histtype='step',
                        label=name,
                        edgecolor=fold_colors.get(fold, "gray"),
                        facecolor=fold_colors.get(fold, "gray"),
                        linewidth=category[cat].get("linewidth", 1),
                        linestyle=category[cat].get("linestyle", "-"),
                    )
                    # ratio plot
                    ratio_draw_config = {
                        'ratio_color': {
                            cat: fold_colors.get(fold, "gray") for cat in category.keys()
                        },
                        'linestyle': {
                            cat: category[cat].get("linestyle", "-") for cat in category.keys()
                        },
                        'marker': {
                            cat: category[cat].get("marker", "o") for cat in category.keys()
                        }
                    }

                    draw_ratio_plot(
                        ax_ratio,
                        hist_df=data['hist'],
                        bin_edges=data['bin_edges'],
                        category_map=category,
                        name=name,
                        reference_df=loop_data[classifier][fold_reference][dataset_type]['hist'],
                        ratio_pulls=ratio_pulls,
                        draw_config=ratio_draw_config,
                    )

            handles, labels = ax_main.get_legend_handles_labels()
            sorted_handles_labels = sorted(
                zip(handles, labels), key=lambda hl: category[list(category)[0]]['name'] in hl[1]
            )
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

            ax_main.set_ylabel("A.U.")
            ax_main.legend(sorted_handles, sorted_labels)

            if ratio_pulls:
                ax_ratio.set_ylabel(r"$Pulls [\sigma]$")
            else:
                ax_ratio.set_ylabel(f"Ratio to {fold_reference}")
            ax_ratio.axhline(1, linestyle="--", color="gray")  # Reference line at y=1

            if y_log:
                ax_main.set_yscale('log')
                # ax_ratio.set_yscale('log')

            # Formatting ratio plot
            x_label = "MVA Score"
            ax_main.set_xlabel(x_label)
            ax_ratio.set_xlabel(x_label)

            if save_path is not None:
                fold_fig_path = save_path / 'fold_comparison'
                # make directory if not exists
                fold_fig_path.mkdir(parents=True, exist_ok=True)
                # save png
                fold_fig_path = fold_fig_path / f"{classifier}_{dataset_type}.png"
                fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')
                # save pdf
                fold_fig_path = fold_fig_path.with_suffix('.pdf')
                fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')

            plt.close(fig)

    #############################################
    #                PLOTTING                   #
    #   classifier summary plots for all folds  #
    #############################################
    logger.info("[Score Distribution] --> Plotting comparison plots per fold")
    dataset_type = reference_dataset
    for fold in data_handler.get_fold_names():

        all_empty = all([
            loop_data[classifier][fold][dataset_type]['hist'].empty
            for classifier in data_handler.get_classifier_names()
            for dataset_type in loop_data[classifier][fold].keys()
        ])

        if all_empty:  # skip if all classifiers are empty
            continue

        # plot scores comparison for each fold with all classifiers
        fig, (ax_main, ax_ratio, ax_sig) = plt.subplots(
            3, 1,
            figsize=figsize,
            gridspec_kw={'height_ratios': [3.5, 1, 1], 'hspace': 0.0},
            sharex=True,
        )
        for classifier in data_handler.get_classifier_names():
            data = loop_data[classifier][fold][dataset_type]
            classifier_config = data_handler.classifier_config[classifier]

            if data['hist'].empty:
                continue

            for i, cat in enumerate(category.keys()):
                if i == 0:
                    name = (
                        f"{classifier}: "
                        r"($\sum \sigma^2$"
                        f"= {np.sqrt(sum(sig_results[classifier][fold][dataset_type] ** 2)):^10.3f})"
                    )
                else:
                    name = None
                ax_main.hist(
                    x=(data['bin_edges'][:-1] + data['bin_edges'][1:]) / 2,
                    bins=data['bin_edges'],
                    weights=data['hist'][str(cat)],
                    density=True,
                    alpha=1.0,
                    histtype='step',
                    label=name,
                    edgecolor=classifier_config.get("score_plot_color", "gray"),
                    facecolor=classifier_config.get("score_plot_color", "gray"),
                    linewidth=category[cat].get("linewidth", 1),
                    linestyle=category[cat].get("linestyle", "-"),
                )

                # ratio plot
                if len(data['bin_edges']) != len(loop_data[baseline_classifier][fold][dataset_type]['bin_edges']):
                    logger.error(
                        f"[{fold}] - [{dataset_type}]: Bin edges of {classifier} and {baseline_classifier} do not match. "
                        f"Skipping ratio plot."
                    )
                    continue
                if classifier != baseline_classifier:
                    ratio_draw_config = {
                        'ratio_color': {
                            cat: classifier_config.get("score_plot_color", "gray") for cat in category.keys()
                        },
                        'linestyle': {
                            cat: category[cat].get("linestyle", "-") for cat in category.keys()
                        },
                        'marker': {
                            cat: classifier_config.get("marker", "o") for cat in category.keys()
                        }
                    }

                    draw_ratio_plot(
                        ax_ratio,
                        hist_df=data['hist'],
                        bin_edges=data['bin_edges'],
                        category_map=category,
                        name=name,
                        reference_df=loop_data[baseline_classifier][fold][dataset_type]['hist'],
                        ratio_pulls=ratio_pulls,
                        draw_config=ratio_draw_config,
                    )

            # significance plot
            ax_sig.plot(
                (data['bin_edges'][:-1] + data['bin_edges'][1:]) / 2,  # X-axis
                sig_results[classifier][fold][dataset_type],  # Y-axis
                color=classifier_config.get("score_plot_color", "gray"),
                label=f"{classifier}",
                marker='o',
                linestyle='-',
            )
            ax_sig.set_ylabel(r"Significance [$\sigma$]")

        # draw dummy lines for categories in legend
        for cat, cat_config in category.items():
            ax_main.plot(
                [], [],
                label=cat_config.get("name", cat),
                color='gray',
                linestyle=cat_config.get("linestyle", "-"),
                # linewidth=cat_config.get("linewidth", 1),
                linewidth=1.5,
            )

        ax_main.set_ylabel("A.U.")
        ax_main.legend()

        if ratio_pulls:
            ax_ratio.set_ylabel(r"$Pulls [\sigma]$")
        else:
            ax_ratio.set_ylabel(f"Ratio to {baseline_classifier}")
        ax_ratio.axhline(1, linestyle="--", color="gray")  # Reference line at y=1

        if y_log:
            ax_main.set_yscale('log')
            ax_ratio.set_yscale('log')
            ax_sig.set_yscale('log')

        # Formatting ratio plot
        x_label = "MVA Score"
        ax_main.set_xlabel(x_label)
        ax_ratio.set_xlabel(x_label)
        ax_sig.set_xlabel(x_label)

        if save_path is not None:
            fold_fig_path = save_path / fold
            # make directory if not exists
            fold_fig_path.mkdir(parents=True, exist_ok=True)
            # save png
            if trafo is None:
                fold_fig_path = fold_fig_path / f"comparison.png"
            else:
                fold_fig_path = fold_fig_path / f"comparison_trafo.png"
            fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')
            # save pdf
            fold_fig_path = fold_fig_path.with_suffix('.pdf')
            fig.savefig(fold_fig_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

    for classifier, fold_results in sig_results.items():
        for fold, data in fold_results.items():
            for data_type, sig in data.items():
                if isinstance(sig, np.ndarray):
                    # If sig is an array, calculate the square root of the sum of squares
                    sig_results[classifier][fold][data_type] = np.sqrt(np.sum(sig ** 2))
                else:
                    pass

    return sig_results
