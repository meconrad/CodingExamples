import numpy as np


def fibonacci(n):
    """
    Recursive function to calculate the value of the Fibonacci sequence at n.
    The value at n in the Fibonacci sequence is the sum of the two preceding numbers.

    Parameters
    ----------
    n : int

    Returns
    -------
    val : int
        value of the Fibonacci sequence at n
    """
    if n <= 1:
        return n
    else:
        f = fibonacci(n-1) + fibonacci(n-2)
    return f


def factorial(n):
    """ 
    Return the factorial of n as a recursive function.

    Parameters
    ----------
    n : int
    
    Returns
    -------
    val_at_n : int
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
