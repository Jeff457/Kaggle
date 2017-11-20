# useful methods go here
# for instance, generating lists of parameter values


def gen_params(start, stop, step):
    """
    :param start: minimum value of parameter
    :param stop: maximum value of parameter
    :param step: amount the generator increments by
    :return: list of parameters within the interval [start, stop)
    """
    return [p for p in range(start, stop, step)]
