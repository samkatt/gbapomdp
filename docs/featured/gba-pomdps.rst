===================
Featured GBA-POMDPs
===================

The GBA-POMDP is a POMDP of which the state space is the intersection of the
(original) POMDP domain state and models of its dynamics. Those models can be
represented and updated in different ways, such as neural networks or tables.
Here we provide those implementations:

.. autosummary::

   general_bayes_adaptive_pomdps.models.tabular_bapomdp
   general_bayes_adaptive_pomdps.models.baddr
   general_bayes_adaptive_pomdps.partial_models.partial_gbapomdp
