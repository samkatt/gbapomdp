""" sample functionality """


def sample_n_unique(sampling_f: callable, num: int) -> list:
    """ samples n **unique** instances using

    Args:
         sampling_f: (`callable`): the sampling function (sampling_f() is called to sample)
         num: (`int`): number of **unique** samples

    RETURNS (`list`): a list of samples

    """

    res = []
    while len(res) < num:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def sample_n(sampling_f: callable, num: int) -> list:
    """ samples n **non-unique** instances

    Assumes: sampling_f is a function that can be called and returns comparable objects
    Note: can return duplicates

    Args:
         sampling_f: (`callable`): the function to call to sample
         num: (`int`): the amount of samples

    RETURNS (`list`): list of samples

    """

    res = []
    while len(res) < num:
        res.append(sampling_f())
    return res
