def reverse_string(string_input):
    """
    Reverse the objects contained in the string
    
    Parameters
    ----------
    string_input : str
    
    Returns
    -------
    reversed_string : str
        the characters of string_input returned in reverse order
    """
    reversed_string = string_input[::-1]
    return reversed_string


def check_for_palindrome(input_string):
    """
    Check if the characters of a string are the same forwards and backwards.
    Not necessarily a palindrome if whitespace included.

    Parameters
    ----------
    string_input : str

    Returns
    -------
    bool
    """
    return input_string == input_string[::-1]
