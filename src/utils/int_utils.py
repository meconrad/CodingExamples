import numpy as np


def sum_to_target(sample_array, target):
    """ 
    Returns the indices of two numbers that add up to a specific target.

    Parameters
    ----------
    sample_array : np.array

    target : int
    
    Returns
    -------
    indices : tuple
    """

    for i, item in enumerate(sample_array):
        remainder = target - item
        if remainder in sample_array:
            i2 = np.where(sample_array == remainder)[0][0]
            break
    if i != i2:
        return (i, i2)
    else:
        return None