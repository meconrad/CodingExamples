import numpy as np
from src.utils.int_utils import sum_to_target


def test_sum_to_target_easy_array():
    """ 
    Test Sum to Target utility for proper output.
    """
    # test an easy array with lots of options
    test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sum_to_target(test_array, 10) == (0, 8)
    assert sum_to_target(test_array, 5) == (0, 3)


def test_sum_to_target_unsorted_array():
    """ 
    Test Sum to Target utility for proper output.
    """
    # test an unsorted array
    test_array = np.array([6, 2, 5, 9, 1, 3, 5, 8])
    assert sum_to_target(test_array, 10) == (1, 7)
    assert sum_to_target(test_array, 5) == (1, 5)


def test_sum_to_target_array_with_zero():
    """ 
    Test Sum to Target utility for proper output.
    """
    #
    test_array = np.array([10, 0, 6, 3, 5, 8])
    assert sum_to_target(test_array, 10) == (0, 1)
    assert sum_to_target(test_array, 5) == (1, 4)


def test_sum_to_target_array_with_no_options():
    """ 
    Test Sum to Target utility for proper output.
    """
    # No 2 objects sum to 12 without duplicating a single object
    test_array = np.array([10, 0, 6, 3, 5, 8])
    assert sum_to_target(test_array, 12) == None