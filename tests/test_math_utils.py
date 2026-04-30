from src.utils.math_utils import fibonacci, factorial


def test_fibonacci():
    """ 
    Test Fibonacci utility for proper output.
    """
    expected_output = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    output = [fibonacci(i) for i in range(10)]
    assert output == expected_output


def test_factorial():
    """
    Test Factorial utility for proper output.
    """
    expected_output = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    output = [factorial(i) for i in range(10)]
    assert output == expected_output
