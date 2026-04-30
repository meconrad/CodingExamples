from src.utils.string_utils import reverse_string, check_for_palindrome


def test_reverse_string():
    """ 
    Test Reverse String utility for proper output.
    """
    test_string = "Code Testing is Best Practice!"
    expected_output1 = "!ecitcarP tseB si gnitseT edoC"
    output1 = reverse_string(test_string)
    assert output1 == expected_output1

    output2 = reverse_string(output1)
    assert output2 == test_string


def test_check_for_palindrome():
    """ 
    Test Check For Palindrome utility for proper output.
    """
    output = check_for_palindrome("madam")
    assert output is True

    output = check_for_palindrome("12321")
    assert output is True

    output = check_for_palindrome("Code Testing is Best Practice!")
    assert output is False
