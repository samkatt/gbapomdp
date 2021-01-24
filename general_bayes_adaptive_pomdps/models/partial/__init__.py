"""Package that contains partial GBA-POMDPs

We call these GBA-POMDPs partial because they do not _completely_ follow the
formal definition. The formal definition is not very forgiving or flexible for
all types of prior knowledge. For example, it could be challenging to capture a
prior where dynamics of a subset of the state features are known and need not
be updated.

This package contains both general structures that simplify creating such
GBA-POMDPs:

    - :mod:`general_bayes_adaptive_pomdps.partial.partial_gbapomdp`

and the specific implementations:

    - :package:`general_bayes_adaptive_pomdps.partial.domain`
"""
