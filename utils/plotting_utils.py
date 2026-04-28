import numpy as np
import matplotlib.pyplot as plt


def create_multi_fig_with_stats(
        input_data: dict,
        plot_type: str,
        show_fig: bool=True
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

    show_fig : bool
        Show the figure when running the function

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    ncols = 2
    nrows = len(input_data) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3*nrows))
    for i, (ax, (variable, dataset)) in enumerate(zip(axes.flatten(), input_data.items())):
        ax.grid(True, alpha=0.5)
        match plot_type:
            case "hist":
                ax.hist(dataset, bins=100, color='darkgrey')
                ax.axvline(np.median(dataset), color='k', ls='--', lw=1, label="median")
                ax.axvline(np.mean(dataset), color='#D55E00', ls='--', lw=1, label="mean")
                ax.axvline(np.mean(dataset) + 3.0 * np.std(dataset), color='g', ls='--', lw=1, label=r"+3$\sigma$")
                ax.axvline(np.mean(dataset) - 3.0 * np.std(dataset), color='b', ls='--', lw=1, label=r"-3$\sigma$")
            case "scatter":
                ax.plot(dataset, 'o', ms=1, color='darkgrey')
                ax.axhline(np.median(dataset), color='k', ls='--', lw=1, label="median")
                ax.axhline(np.mean(dataset), color='#D55E00', ls='--', lw=1, label="mean")
                ax.axhline(np.mean(dataset) + 3.0 * np.std(dataset), color='g', ls='--', lw=1, label=r"+3$\sigma$")
                ax.axhline(np.mean(dataset) - 3.0 * np.std(dataset), color='b', ls='--', lw=1, label=r"-3$\sigma$")

        ax.set_title(variable)
        if i == 1:
            ax.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
    plt.tight_layout()
    if show_fig is True:
        plt.show()
    return fig
