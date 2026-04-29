from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import numpy as np

from utils.plotting_utils import plot_gmm_results


def identify_multi_modes_in_dataset(
        multimodal_data: list,
        max_components: int,
        max_iter: int=1000,
        show_fig: bool=True
        ):
    """ 

    Parameters
    ----------
    multimodal_data : list
        Dataset to test with GMM

    max_components : int
        Max components to test with GMM

    max_iter : int, default = 1000
        Max iterations to supply to GMM

    show_fig : bool, default = True
        Display the output in a figure

    Returns
    -------
    """
    # Set up data for GMM and plotting
    X = np.array(multimodal_data).reshape(-1, 1)
    x = np.linspace(X.min()-2, X.max()+2, 1000).reshape(-1, 1)

    # Repeat GMM fitting iteratively up to max_components, then identify best-fit
    gmm_results = {}
    for n_components in range(1, max_components):
        gmm_results[n_components] = {}
        GMM = GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            init_params="kmeans",
            random_state=0,
            )
        gmm = GMM.fit(X)

        # Add fit results to gmm_results dict
        gmm_results[n_components]["means"] = gmm.means_.flatten()
        gmm_results[n_components]["covariances"] = gmm.covariances_.flatten()
        gmm_results[n_components]["weights"] = gmm.weights_.flatten()

        # Get the Bayesian Information Criterion - lowest score will be the best fit
        gmm_results[n_components]["bic"] = gmm.bic(X)

        # Total PDF: Weighted sum of individual PDFs
        # score_samples returns log-likelihood, exp() converts to probability
        log_probs = gmm.score_samples(x)
        gmm_results[n_components]["pdf_total"] = np.exp(log_probs)

        # Get Individual Component PDFs
        pdf_comps = []
        for i, _ in enumerate(gmm_results[n_components]["weights"]):
            pdf_comps.append(
                gmm_results[n_components]["weights"][i] * stats.norm.pdf(
                    x,
                    gmm_results[n_components]["means"][i],
                    np.sqrt(gmm_results[n_components]["covariances"][i])
                    )
                )
        gmm_results[n_components]["pdf_components"] = pdf_comps

        # Get score
        gmm_results[n_components]["log_likelihood"] = round(gmm.score(X), 3)

    # find the lowest Bayesian Information Criterion
    d = {k: v["bic"] for k, v in gmm_results.items()}
    best_fit_component = min(d, key=d.get)

    # calculate upper and lower thresholds for each mode component
    upper = gmm_results[best_fit_component]["means"] + 3.0 * np.sqrt(gmm_results[best_fit_component]["covariances"])
    lower = gmm_results[best_fit_component]["means"] - 3.0 * np.sqrt(gmm_results[best_fit_component]["covariances"])

    # Plotting the GMM fitted components and their associated 3-sigma ranges for the best fit, if requested
    fig = None
    if show_fig is True:
        fig = plot_gmm_results(
            X,
            gmm_results,
            best_fit_component,
            upper,
            lower
        )

    return gmm_results, best_fit_component, upper, lower, fig

