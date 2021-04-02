=====
Usage
=====

To use General Bayes-adaptive POMDPs in a project::

    import general_bayes_adaptive_pomdps

This package provides some well-known POMDPs out of the box, of which
GBA-POMDPs can be formulated, for example::

    
    # create a POMDP 
    # (of which the dynamics are not necessarily implemented)
    env = create_domain(
        "tiger",
        0,
        use_one_hot_encoding=True,
    )

    # create BADDr:
    # a GBA-POMDP that uses dropout-networks as dynamics posterior
    baddr = BADDr(
        env,
        num_nets,
        optimizer,
        learning_rate,
        network_size,
        batch_size
    )

    # create a prior training method
    train_method = create_train_method(
        prior_certainty, prior_correctness, num_epochs, batch_size
    )

    # reset BADDr (pre-trains networks
    baddr.reset(train_method, learning_rate, online_learning_rate)

In this example `baddr` can then be used as a regular POMDP to do planning in.

.. toctree::
    :maxdepth: 1
    :caption: Guides

    guides/builtin-domains
