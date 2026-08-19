import numpy as np


def trafoD_binning(test_data, test_label, test_weights, min_bkg_per_bin=3,
                   reweight_factor=1, trafoSixY=10, trafoSixZ=10, trafoSixtyMCLowBound=3, trafoSixMCstatUpBound=1):
    """
    Optimized TrafoD binning function closely aligned with the C++ implementation.

    Parameters:
    - test_data: np.array, array of data values (e.g., classifier scores).
    - test_label: np.array, binary labels (0 for background, 1 for signal).
    - test_weights: np.array, weights for each event.
    - Zb: float, weight factor for background events in Z calculation.
    - Zs: float, weight factor for signal events in Z calculation.
    - min_bkg_per_bin: int, minimum number of background events per bin.
    - reweight_factor: float, reweight factor for signal events.
    - trafoSixY: float, signal weight factor for bin merging.
    - trafoSixZ: float, background weight factor for bin merging.
    - trafoSixtyMCLowBound: float, minimum MC events required per bin.
    - trafoSixMCstatUpBound: float, upper bound for relative MC stats uncertainty.

    Returns:
    - bin_edges: list of bin edges satisfying the TrafoD criteria.
    """

    bin_edges = []
    total_events = len(test_data)

    # Sort data by the test data values for proper binning
    sorted_indices = np.argsort(test_data)
    data_sorted = test_data[sorted_indices]
    label_sorted = test_label[sorted_indices]
    weights_sorted = test_weights[sorted_indices]

    weights_sorted[label_sorted == 1] /= reweight_factor

    # Precompute total signal and background weights
    N_b = np.sum(weights_sorted[label_sorted == 0])
    N_s = np.sum(weights_sorted[label_sorted == 1])

    # Compute cumulative sums for signal and background counts
    cumulative_weights = np.cumsum(weights_sorted)
    cumulative_signal = np.cumsum(weights_sorted * (label_sorted == 1))
    cumulative_background = np.cumsum(weights_sorted * (label_sorted == 0))

    # Initialize bin starting point
    bin_start = 0

    while bin_start < total_events:
        dist_prev = float("inf")  # **Added: Initialize dist_prev for distance comparison**

        for bin_end in range(bin_start + 1, total_events + 1):
            n_s = cumulative_signal[bin_end - 1] - (cumulative_signal[bin_start - 1] if bin_start > 0 else 0)
            n_b = cumulative_background[bin_end - 1] - (cumulative_background[bin_start - 1] if bin_start > 0 else 0)

            # TrafoD formula for relative uncertainty
            err2Rel = (1 / ((n_b / (N_b / trafoSixZ)) + (n_s / (N_s / trafoSixY)))) if n_b > 0 and n_s > 0 else float(
                "inf")
            dist = abs(err2Rel - 1)  # **Added: Distance metric**

            # Apply TrafoD criteria (with additional MC protection)
            if (np.sqrt(err2Rel) < 1 and
                    n_b >= min_bkg_per_bin and
                    (n_b + n_s) >= trafoSixtyMCLowBound and
                    np.sqrt(n_b / (n_b + n_s)) < trafoSixMCstatUpBound):

                if dist < dist_prev:  # **Changed: Only proceed if distance improves**
                    dist_prev = dist
                    continue

                bin_edges.append(data_sorted[bin_end - 1])  # Set bin edge
                bin_start = bin_end  # Move to next starting point
                break
        else:
            break  # Stop if no valid bin end is found

    # Finalize bin edges and handle merging if necessary
    bin_edges.append(data_sorted[-1])

    # **Added: Merge left-most bins if necessary**
    if len(bin_edges) > 1:
        sum_sig_lowest = cumulative_signal[bin_start - 1] if bin_start > 0 else cumulative_signal[-1]
        sum_bkg_lowest = cumulative_background[bin_start - 1] if bin_start > 0 else cumulative_background[-1]

        sum_sig_second_lowest = cumulative_signal[-2]
        sum_bkg_second_lowest = cumulative_background[-2]

        if abs((sum_sig_lowest + sum_bkg_lowest) / (sum_sig_second_lowest + sum_bkg_second_lowest) - 1) > 0.5:
            bin_edges.pop(-2)  # Merge bins

    # **Added: Ensure total number of bins does not exceed Y+Z**
    if len(bin_edges) > trafoSixY + trafoSixZ + 1:
        bin_edges = bin_edges[:trafoSixY + trafoSixZ + 1]

    return bin_edges


def get_rebin_bins(histo_bkg, histo_sig, histo_bkg_unc, histo_sig_unc, max_unc, trafo_sixty_mc_low_bound,
                   trafo_six_y, trafo_six_z, trafo_six_mc_stat_up_bound):
    """
    Auto-binning function for the trafo60 method, now incorporating background and signal uncertainties.

    Parameters:
        histo_bkg (numpy.ndarray): Background histogram values.
        histo_sig (numpy.ndarray): Signal histogram values.
        histo_bkg_unc (numpy.ndarray): Uncertainties of the background histogram.
        histo_sig_unc (numpy.ndarray): Uncertainties of the signal histogram.
        max_unc (float): Maximum allowed uncertainty.
        trafo_sixty_mc_low_bound (float): Minimum MC events required per bin.
        trafo_six_y (float): Parameter for signal normalization.
        trafo_six_z (float): Parameter for background normalization.
        trafo_six_mc_stat_up_bound (float): Upper bound for MC statistical uncertainty.

    Returns:
        list: Bin edges where rebinned.
    """
    # Initialize variables
    n_bins = len(histo_bkg)
    bins = [n_bins]  # Start with overflow bin index

    sum_bkg, sum_sig, err2_bkg, err2_sig = 0, 0, 0, 0
    for i_bin in range(n_bins, 0, -1):
        n_bkg_bin = histo_bkg[i_bin - 1]
        n_sig_bin = histo_sig[i_bin - 1] if histo_sig is not None else 0
        unc_bkg_bin = histo_bkg_unc[i_bin - 1] if histo_bkg_unc is not None else 0
        unc_sig_bin = histo_sig_unc[i_bin - 1] if histo_sig_unc is not None else 0

        sum_bkg += n_bkg_bin
        sum_sig += n_sig_bin
        err2_bkg += unc_bkg_bin ** 2
        err2_sig += unc_sig_bin ** 2

        # Calculate relative uncertainty using provided uncertainties
        err2_rel_bkg = err2_bkg / sum_bkg ** 2 if sum_bkg > 0 else float('inf')
        err2_rel_sig = err2_sig / sum_sig ** 2 if sum_sig > 0 else float('inf')

        err2_rel = (
            1 / (sum_bkg / (np.sum(histo_bkg) / trafo_six_z) + sum_sig / (np.sum(histo_sig) / trafo_six_y))
            if sum_bkg > 0 and sum_sig > 0 else float('inf')
        )

        # Check conditions for bin merging
        if (
                np.sqrt(err2_rel) < 1
                and np.sqrt(err2_rel_bkg) < max_unc
                and sum_bkg + sum_sig >= trafo_sixty_mc_low_bound
                and np.sqrt(err2_rel_bkg) < trafo_six_mc_stat_up_bound
                and np.sqrt(err2_rel_sig) < trafo_six_mc_stat_up_bound  # Ensure signal uncertainty is within bounds
        ):
            bins.append(i_bin - 1)

            # Reset variables for the next bin
            sum_bkg, sum_sig, err2_bkg, err2_sig = 0, 0, 0, 0

    bins.append(0)  # Add underflow bin index

    # Ensure bins are in descending order
    bins = sorted(bins, reverse=True)

    # Merge the last two bins if the condition is met
    if len(bins) > 2:
        sum_bkg_lowest = np.sum(histo_bkg[: bins[-2]])
        sum_sig_lowest = np.sum(histo_sig[: bins[-2]]) if histo_sig is not None else 0
        sum_bkg_2nd_low = np.sum(histo_bkg[bins[-2]: bins[-1]])
        sum_sig_2nd_low = np.sum(histo_sig[bins[-2]: bins[-1]]) if histo_sig is not None else 0

        if abs((sum_sig_lowest + sum_bkg_lowest) / (sum_sig_2nd_low + sum_bkg_2nd_low + 1e-6) - 1) > 0.5:
            bins.pop(-2)

    return bins


def convert_indices_to_edges(bin_indices, original_bin_edges):
    """
    Convert bin indices to actual bin edges.

    Parameters:
        bin_indices (list): List of bin indices, including overflow and underflow.
        original_bin_edges (numpy.ndarray): Array of original bin edges.

    Returns:
        numpy.ndarray: Array of rebinned edges.
    """
    # Reverse the bin indices to maintain ascending order of edges
    bin_indices = sorted(bin_indices)

    # Map the indices to the corresponding bin edges
    rebinned_edges = [original_bin_edges[i] for i in bin_indices]

    return np.array(rebinned_edges)


def convert_indices_to_edges_under_overflow(bin_indices, original_bin_edges):
    """
    Convert bin indices to actual bin edges, considering underflow and overflow bins.

    Parameters:
        bin_indices (list): List of bin indices, including overflow and underflow.
        original_bin_edges (numpy.ndarray): Array of original bin edges (without underflow/overflow).

    Returns:
        numpy.ndarray: Array of rebinned edges.
    """
    # Extend bin edges to account for underflow and overflow bins
    extended_bin_edges = np.concatenate((
        # [original_bin_edges[0] - (original_bin_edges[1] - original_bin_edges[0])],  # Underflow
        original_bin_edges,
        [original_bin_edges[-1] + (original_bin_edges[-1] - original_bin_edges[-2])]  # Overflow
    ))

    rebinned_edges = [original_bin_edges[-1] + (original_bin_edges[-1] - original_bin_edges[-2])]
    # Map the indices to the corresponding bin edges
    rebinned_edges = rebinned_edges + [extended_bin_edges[i - 1 if i > 0 else 0] for i in bin_indices[1:]]
    return np.array(rebinned_edges)


def calculate_binned_significance(
        h_s, h_b, h_b_unc=None,
        method="asimov", sig_scale: float = 1.0,
        bkg_scale: float = 1.0
):
    """
    Calculate the binned significance for signal and background histograms, ignoring NaN values.

    Parameters:
        h_s (numpy.ndarray): Signal histogram values (rebinned).
        h_b (numpy.ndarray): Background histogram values (rebinned).
        method (str): Method to compute significance ("poisson" or "asimov").

    Returns:
        numpy.ndarray: Binned significance values.
    """
    # Replace NaN with 0 in signal and background
    h_s = np.nan_to_num(h_s, nan=0.0) * sig_scale
    h_b = np.nan_to_num(h_b, nan=0.0) * bkg_scale

    # Initialize significance array
    significance = np.zeros_like(h_s, dtype=float)

    if method == "poisson":
        # Simple Poisson significance
        significance = np.where(h_b > 0, h_s / np.sqrt(h_b), 0.0)

    elif method == "asimov":
        if h_b_unc is None:
            raise ValueError("Background uncertainty (h_b_unc) is required for Asimov significance with systematics.")

        # Ensure uncertainties are positive and handle invalid values
        h_b_unc = np.nan_to_num(h_b_unc, nan=0.0)
        h_b_unc = np.maximum(h_b_unc, 0.0)

        # Compute Asimov significance formula
        n = h_s + h_b  # Total expected events per bin
        b2 = h_b ** 2
        sigma2 = h_b_unc ** 2

        valid = (h_b > 0) & (h_s > 0) & (sigma2 >= 0)

        term1 = np.zeros_like(h_b, dtype=np.float64)
        term2 = np.zeros_like(h_b, dtype=np.float64)

        term1[valid] = n[valid] * np.log(
            (n[valid] * (h_b[valid] + sigma2[valid])) / (b2[valid] + n[valid] * sigma2[valid]))
        numerator = sigma2[valid] * (n[valid] - h_b[valid])
        denominator = h_b[valid] * (h_b[valid] + sigma2[valid])
        term2[valid] = (b2[valid] / sigma2[valid]) * np.log(1 + numerator / denominator)

        term = 2 * (term1 - term2)
        term = np.maximum(term, 0)  # Ensure non-negative values

        significance = np.sqrt(term)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'poisson' or 'asimov'.")

    return significance


def root_to_numpy_histogram(histogram):
    # Get the number of bins
    bins = histogram.GetNbinsX()

    # Create a numpy histogram
    numpy_histogram = np.zeros(bins)

    # Fill the numpy histogram with the ROOT histogram data
    for i in range(bins):
        numpy_histogram[i] = histogram.GetBinContent(
            i + 1)  # Fix the index to access the elements of the numpy histogram

    return numpy_histogram
