import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils.plotting_utils import plot_gaussian_mixture_data


random.seed(45)


def create_multi_modal_test_data(
        loc_range: tuple = (0, 100),
        std_range: tuple = (1, 5),
        size_range: tuple = (500, 2000),
        n_modes: int = 3,
        noise_std: float = 5.0,
        noise_size: int = 100,
        show_fig: bool = True,
):
    """ 
    Generate fake test data with multiple modes.
    
    Parameters
    ----------
    loc_range : tuple, default = (0, 100)
        (minimum dataset value, maximum dataset value)

    std_range : tuple, default = (1, 5)
        (minimum std dev value, maximum std dev value)

    size_range : tuple, default = (500, 2000)
        (minimum loc value, maximum loc value)

    n_modes : int, default = 3
        the number of modes to generate

    noise_std : float, default = 5.0
        standard deviation of noise

    noise_size : int, default = 100
        number of "noisy" datapoints to add to distribution

    show_fig : bool, default = True
        plot multimodal data and individual modes on figure

    Returns
    -------
    data : list
        shuffled list of multi-modal datapoints

    modes : list of list
        multi-modal datasets individualized

    fig : matplotlib.figure.Figure

    """
    modes = []
    # For each mode requested, randomly choose a location, scale, and dataset size
    # then add noise, add dataset to list of modes
    for _ in range(n_modes):
        loc = random.randint(*loc_range)
        data_std = random.randint(*std_range)
        data_size = random.randint(*size_range)
        mode_data = (
            list(stats.norm.rvs(loc=loc, scale=data_std, size=data_size)) + 
            list(stats.norm.rvs(loc=loc, scale=noise_std, size=noise_size))
        )
        modes.append(mode_data)

    # add all modes to single dataset, then shuffle
    data = modes[0] + modes[1] + modes[2]
    random.shuffle(data)

    # Generate and show results on a figure if requested
    fig = None
    if show_fig is True:
        fig = plot_gaussian_mixture_data(data, modes)

    return data, modes, fig


def create_test_data(
        noise_std: float = 5.0,
        noise_size: int = 100,
):
    """ 
    Generate fake test data for a variety of commonly used distributions.
    
    Parameters
    ----------
    noise_std : float, default = 5.0
        standard deviation of noise

    noise_size : int, default = 100
        number of "noisy" datapoints to add to distribution

    Returns
    -------
    data : dict of list
        keys = title of dataset, values = list of values
        {
            "gamma": [...],
            "normal": [...],
            "beta": [...],
            "exponential": [...],
            "lognormal": [...],
            "triang": [...]
            }
    """
    # create each distribution and add noise, then shuffle datasets
    data = {
        "gamma": list(stats.gamma.rvs(a=3.0, loc=1.5, scale=2.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
        "normal": list(stats.norm.rvs(loc=1.5, scale=2.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
        "beta": list(stats.beta.rvs(a=3.0, b=2.0, loc=1.5, scale=2.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
        "exponential": list(stats.expon.rvs(loc=1.5, scale=2.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
        "lognormal": list(stats.lognorm.rvs(s=0.5, loc=1.5, scale=5.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
        "triang": list(stats.triang.rvs(c=1.0, loc=1.5, scale=2.0, size=1000)) + list(stats.norm.rvs(loc=0, scale=noise_std, size=noise_size)),
    }

    # Shuffle the noise in with the datasets so noise is more natural
    for dataset in data.values():
        random.shuffle(dataset) 

    return data