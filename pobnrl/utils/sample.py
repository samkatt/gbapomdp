""" sample functionality """


def sample_n_unique(sampling_f, num) -> list:
    """sample_n_unique samples n **unique** instances using

    :param sampling_f: the sampling function (sampling_f() is called to sample)
    :param num: number of **unique** samples

    :return: a list of samples
    """

    res = []
    while len(res) < num:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def sample_n(sampling_f, num) -> list:
    """sample_n samples n **non-unique** instances

    Assumes: sampling_f is a function that can be called and returns comparable objects
    Note: can return duplicates

    :param sampling_f: the function to call to sample
    :param num: the amount of samples
    :rtype: list of samples
    """

    res = []
    while len(res) < num:
        res.append(sampling_f())
    return res
