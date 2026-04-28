import numpy as np


def sum_to_target(sample_array, target):
    """ 
    Returns the indices of two numbers that add up to a specific target in an array.

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
    return (i, i2)