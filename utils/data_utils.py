import numpy as np
import scipy.stats as stats
import random

random.seed(45)


def create_multi_modal_test_data(
        n_modes: int = 3,
        noise_std: float = 5.0,
        noise_size: int = 100,
):
    """ 
    Generate fake test data with multiple modes.
    
    Parameters
    ----------
    n_modes: int
        the number of modes to generate, default = 3

    noise_std : float
        standard deviation of noise, default = 5.0

    noise_size : int
        number of "noisy" datapoints to add to distribution, default = 100

    Returns
    -------
    data : list
        shuffled list of multi-modal datapoints

    modes: list of list
        multi-modal datasets individualized

    """
    modes = []
    # for each mode requested, randomly choose a location, scale, and dataset size
    # then add noise, add dataset to list of modes
    for _ in range(n_modes):
        loc = random.randint(0, 30)
        data_std = random.randint(1, 5)
        data_size = random.randint(500, 2000)
        mode_data = (
            list(stats.norm.rvs(loc=loc, scale=data_std, size=data_size)) + 
            list(stats.norm.rvs(loc=loc, scale=noise_std, size=noise_size))
        )
        modes.append(mode_data)

    # add all modes to single dataset, then shuffle
    data = modes[0] + modes[1] + modes[2]
    random.shuffle(data)

    return data, modes


def create_test_data(
        noise_std: float = 5.0,
        noise_size: int = 100,
):
    """ 
    Generate fake test data for a variety of commonly used distributions.
    
    Parameters
    ----------
    noise_std : float
        standard deviation of noise, default = 5.0

    noise_size : int
        number of "noisy" datapoints to add to distribution, default = 100

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

    for dataset in data.values():
        random.shuffle(dataset) 

    return data