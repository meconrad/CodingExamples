from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import numpy as np

from CodingExamples.src.utils.plotting_utils import plot_gmm_results, plot_gmm_anomaly_detection_results

def percentile_mask(
        data: np.ndarray,
        min_mask: float = 5,
        max_mask: float = 95
        ):
    """ 

    Parameters
    ----------
    data : numpy.ndarray

    min_mask : float, default = 5
        minimum percentile to mask dataset with

    max_mask : float, default = 95
        maximum percentile to mask dataset with

    Returns
    -------
    masked_data : 
    """
    # Calculate percentile values
    upper_pctl = np.percentile(data, max_mask)
    lower_pctl = np.percentile(data, min_mask)
    # Mask dataset / remove datapoints outside these values
    masked_data = data[(lower_pctl < data) & (data < upper_pctl)]
    return masked_data
    

def identify_multi_modes_in_dataset(
        multimodal_data: list,
        min_scrub: float,
        max_scrub: float,
        max_components: int,
        max_iter: int=1000,
        show_fig: bool=True
        ):
    """ 

    Parameters
    ----------
    multimodal_data : list
        Dataset to test with GMM

    min_scrub : float
        minimum percentile to scrub outliers to, i.e. 0.03%

    max_scrub : float
        maximum percentile to scrub outliers to, i.e. 99.7%

    max_components : int
        Max components to test with GMM

    max_iter : int, default = 1000
        Max iterations to supply to GMM

    show_fig : bool, default = True
        Display the output in a figure

    Returns
    -------
    gmm_results : dict of dict
        output from Gaussian Mixture Model implemented in utils.modelling_utils.identify_multi_modes_in_dataset

    best_fit_component : int
        the component with the lowest Bayesian Information Criterion score (BIC)

    upper : np.ndarray
        the mean + 3 sigma threshold of each component

    lower : np.ndarray
        the mean - 3 sigma threshold of each component

    fig : fig : matplotlib.figure.Figure

    """
    # Scrub dataset for outliers
    data = percentile_mask(np.array(multimodal_data), min_mask=min_scrub, max_mask=max_scrub)

    # Set up data for GMM and plotting
    X = np.array(data).reshape(-1, 1)
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
            reg_covar=1e-5
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


def identify_anomalies_in_multimodal_data(
        data: list,
        lower: list,
        upper: list,
        show_fig: bool=True
        
):
    """
    Check every datapoint against 3-sigma thresholds to identify statistical anomalies.

    Parameters
    ----------
    data : list
        list of datapoints to check against thresholds
    
    lower : list
        list of lower thresholds for all components
    
    upper : list
        list of upper thresholds for all components

    show_fig : bool, default = True
        plot data and anomalies with acceptable bands

    Returns
    -------
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

    fig : matplotlib.figure.Figure or None

    """
    # Zip the thresholds for easier comparison
    thresholds = list(zip(
        [round(float(val), 3) for val in lower],
        [round(float(val), 3) for val in upper]
        ))

    # Check each datapoint against all threshold sets
    anomalies = []
    for i, point in enumerate(data):
        check = sum(threshold[0] < point < threshold[1] for threshold in thresholds)
        if check == 0:
            anomalies.append(
                {
                    "date": i, 
                    "value": round(float(point), 3),
                }
            )

    # Build output
    output = {
        "thresholds": thresholds,
        "n_anomalies": len(anomalies),
        "anomaly_ratio": round(len(anomalies)/len(data), 3),
        "anomalies": anomalies
    }

    # Plot results, if requested
    fig = None
    if show_fig is True:
        fig = plot_gmm_anomaly_detection_results(
            data,
            thresholds,
            output
        )

    return output, fig

