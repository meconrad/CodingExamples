import numpy as np
import scipy.stats as stats
import random


def create_test_data(
        noise_std: float = 5.0,
        noise_size: int = 100,
):
    """ 
    
    Parameters
    ----------
    noise_std : float
        standard deviation of noise

    noise_size : int
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