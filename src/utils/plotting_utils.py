import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from typing import Any, List, Dict


# Define accessible colors (e.g., Wong palette)
cb_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Create a cycler that changes both color and linestyle
custom_cycler = (cycler(color=cb_colors) + 
                 cycler(linestyle=["-", "--", ":", "-.", "-", "--", ":"]))


def create_multi_fig_with_stats(
        input_data: dict[List],
        plot_type: str
        ):
    """
    For each variable in input_data, plot histogrammed dataset on a new axes within the larger figure space,
    label legend entries, plot mean, median, and mean +/- 3 sigma thresholds, and add a grid for easy inspection.

    Parameters
    ----------
    input_data : dict of list
        keys = title of dataset, values = list of values
        {
            "variable_1": [1, 2, 3],
            "variable_2": [4, 5, 6],
            "variable_3": [7, 8, 9]
            }
    plot_type : str
        ["hist", "scatter"]

    show_fig : bool, default = True
        Show the figure when running the function

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    ncols = 2
    nrows = len(input_data) // ncols
    # auto-scale figure based on the number of rows and cols needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3*nrows))
    for i, (ax, (variable, dataset)) in enumerate(zip(axes.flatten(), input_data.items())):
        ax.grid(True, alpha=0.5)
        match plot_type:
            case "hist":
                # plot the histogram and the associated stats as vertical lines
                ax.hist(dataset, bins=100, color="darkgrey")
                ax.axvline(np.median(dataset), color="k", ls="--", lw=1, label="median")
                ax.axvline(np.mean(dataset), color="#D55E00", ls="--", lw=1, label="mean")
                ax.axvline(np.mean(dataset) + 3.0 * np.std(dataset), color="g", ls="--", lw=1, label=r"+3$\sigma$")
                ax.axvline(np.mean(dataset) - 3.0 * np.std(dataset), color="b", ls="--", lw=1, label=r"-3$\sigma$")

            case "scatter":
                # plot the scatterplot and the associated stats as horizontal lines
                ax.plot(dataset, "o", ms=1, color="darkgrey")
                ax.axhline(np.median(dataset), color="k", ls="--", lw=1, label="median")
                ax.axhline(np.mean(dataset), color="#D55E00", ls="--", lw=1, label="mean")
                ax.axhline(np.mean(dataset) + 3.0 * np.std(dataset), color="g", ls="--", lw=1, label=r"+3$\sigma$")
                ax.axhline(np.mean(dataset) - 3.0 * np.std(dataset), color="b", ls="--", lw=1, label=r"-3$\sigma$")

        ax.set_title(variable)
        # Allow only one legend and move outside of the axes frame
        if i == 1:
            ax.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
    plt.tight_layout()
    plt.show()
    return fig


def plot_gaussian_mixture_data(
        data: list,
        modes: list
        ):
    """ 
    Plot the multi-modal dataset as well as the individual components.

    Parameters
    ----------
    data : list
        list of datapoints representing the combined components

    modes : list of list
        list of list of datapoints

    Returns
    -------
    fig : matplotlib.figure.Figure
    
    """
    # Set up the figure
    fig = plt.figure(figsize=(7, 3))
    plt.gca().set_prop_cycle(custom_cycler)
    plt.grid(True, alpha=0.4)
    # Add the multimodal dataset
    plt.hist(data, bins=100, label="Multimodal Data")
    # Add the components
    for i, _ in enumerate(modes):
        plt.hist(modes[i], bins=30, alpha=0.5, label=f"Component {i}")
    plt.legend()
    plt.title("Original Generated Multimodal Dataset")
    plt.figtext(
        0.5,
        0.005,
        "*Note: modal data might not fit multimodal dataset exactly due to bin size differences",
        ha="center",
        fontsize=8,
        style="italic"
        )
    plt.show()
    return fig


def plot_gmm_results(
        X: np.ndarray,
        gmm_results: dict[Dict],
        component: int,
        upper: np.ndarray,
        lower: np.ndarray
):
    """ 
    Plot the results of utils.modelling_utils.identify_multi_modes_in_dataset, including the fitted
    dataset, the total multimodal PDF, the fitted components, and the 3-sigma regions of each component.

    Parameters
    ----------
    X : np.ndarray
        numpy array of datapoints
    
    gmm_results : dict of dict
        output from Gaussian Mixture Model implemented in utils.modelling_utils.identify_multi_modes_in_dataset

    component : int
        the component to be plotted, usually the best fit component

    upper : np.ndarray
        the mean + 3 sigma threshold of each component

    lower : np.ndarray
        the mean - 3 sigma threshold of each component

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    # Define range for plotting
    x = np.linspace(X.min()-2, X.max()+2, 1000).reshape(-1, 1)

    # Set up figure
    fig = plt.figure(figsize=(7, 3))
    plt.gca().set_prop_cycle(custom_cycler)
    plt.grid(True, alpha=0.4)

    # Add 3 sigma span regions
    for color, (i, _) in zip(custom_cycler, enumerate(gmm_results[component]["pdf_components"])):
        plt.axvspan(lower[i], upper[i], alpha=0.4, color=color["color"], label=rf"3$\sigma$ Region {i}")

    # Add multimodal dataset as histogram and PDF
    plt.hist(X, bins=100, density=True, color="darkgrey", label="Data")
    plt.plot(x, gmm_results[component]["pdf_total"], "-k", label="Total GMM PDF", linewidth=2)

    # Add the GMM fitted components
    for color, (i, _) in zip(custom_cycler, enumerate(gmm_results[component]["pdf_components"])):
        plt.plot(x, gmm_results[component]["pdf_components"][i], c=color["color"], ls="--", label=f"Component {i}")
        
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
    plt.title(r"GMM Fitted Components, Total PDF, and 3$\sigma$ Thresholds")
    plt.text(
        np.min(x)*1.1,
        np.max(gmm_results[component]["pdf_total"])*0.9,
        f"Log Likelihood = {gmm_results[component]["log_likelihood"]}"
        )
    plt.show()
    return fig


def plot_gmm_anomaly_detection_results(
        data: list,
        thresholds: list[tuple],
        output: dict[Any]
):
    """
    Plot the anomaly detection results from using GMM component splitting.

    Parameters
    ----------
    data : list
        list of datapoints

    thresholds : list of tuple
        (lower, upper)

    output : dict
        {
            "thresholds": list of tuple,
            "n_anomalies": int,
            "anomaly_ratio": float (3 decimals),
            "anomalies": {
                            "date": int,
                            "value": float
                            }
            }

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    # Set up fig
    fig = plt.figure(figsize=(7, 3))
    plt.gca().set_prop_cycle(custom_cycler)
    plt.grid(True, alpha=0.4)

    # Plot acceptable 3-sigma bands
    for thresh in thresholds:
        plt.axhspan(thresh[0], thresh[1], alpha=0.1, color='#009E73', label="acceptable bands")

    # Plot data
    plt.plot(data, 'o', ms=1, color='k', label="data")

    # Overlay anomalies in different colour
    anom_x = [anom["date"] for anom in output["anomalies"]]
    anom_y = [anom["value"] for anom in output["anomalies"]]
    plt.plot(anom_x, anom_y, 'o', ms=1, c='red', label="anomaly")

    # Remove duplicate entries from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2, bbox_to_anchor=(1.05, 1.0))

    plt.title("3-Sigma Anomalies Detected in Synthetic Multimodal Data")

    plt.show()
    return fig

