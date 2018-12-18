""" sample functions """

def sample_n_unique(sampling_f, num):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < num:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res
