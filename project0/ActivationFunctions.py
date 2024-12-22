def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""

    """ My solution:
    return np.maximum(0, x)
    """

    # Instructor's solution: (same)
    return max(0, x)


def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""

    """ My solution:
    if x <= 0:
        return 0
    else:
        return 1
    """
    # Instructor's solution (same)
    if x <= 0:
        return 0
    else:
        return 1

