=============================
General Bayes-adaptive POMDPs
=============================


.. image:: https://img.shields.io/travis/samkatt/general_bayes_adaptive_pomdps.svg
        :target: https://travis-ci.com/samkatt/general_bayes_adaptive_pomdps

.. image:: https://readthedocs.org/projects/general-bayes-adaptive-pomdps/badge/?version=latest
        :target: https://general-bayes-adaptive-pomdps.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A library to create Bayes-adaptive POMDPs for Bayesian RL


* Free software: MIT license
* Documentation: https://general-bayes-adaptive-pomdps.readthedocs.io.


Features
--------

Models to make POMDPs_ with unknown dynamics into general Bayes-adaptive POMDPs
(GBA-POMDPs). GBA-POMDPs themselves are POMDPs with known dynamics that, when
solved, provide an optimal solution to the exploration-exploitation trade-off
in reinforcement learning. These models cast the reinforcement learning problem
into a planning problem.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. POMDPs: https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
