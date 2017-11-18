import numpy as np

# useful methods go here
# for instance, generating lists of parameter values

def gen_params(min, max, step):
    '''
    :param min: minimum value of parameter
    :param max: maximum value of parameter
    :param step: amount the generator increments by
    :return: list of parameters
    '''
    return [p for p in np.arange(min, max, step)]
