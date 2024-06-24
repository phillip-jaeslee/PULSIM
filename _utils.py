import numbers

def is_number(n):
    if isinstance(n, numbers.Real):
        return n
    else:
        raise TypeError("Must be a real number.")

def is_tuple_of_two_numbers(t):
    m, n = t
    return is_number(m), is_number(n)

def low_high(t):
    two_numbers = is_tuple_of_two_numbers(t)
    return min(two_numbers), max(two_numbers)