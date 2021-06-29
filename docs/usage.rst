=====
Usage
=====

To use General Bayes-adaptive POMDPs in a project::

    import general_bayes_adaptive_pomdps

This package provides some well-known POMDPs out of the box, of which
GBA-POMDPs can be formulated, for example:

.. literalinclude:: ../tests/test_integration.py
   :pyobject: test_tabular_bapomdp

In this example `baddr` can then be used as a regular POMDP to do planning in:

.. literalinclude:: ../tests/test_integration.py
   :pyobject: run_gba_pomdp

.. toctree::
    :maxdepth: 1
    :caption: Featured

    featured/gba-pomdps
    featured/builtin-domains
