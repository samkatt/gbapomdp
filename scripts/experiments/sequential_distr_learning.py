#!/usr/bin/env python

""" visualizes learning of a distribution sequentially """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import random
import sys

import numpy as np
import torch

from po_nrl.agents.neural_networks import Net
import po_nrl.pytorch_api


def main() -> None:
    """ trains a network to learn a distribution

    * specifically train a .85/.15 distribution
    * provides 1 data point at a time

    Args:

    RETURNS (`None`):

    """

    conf = parse_arguments()

    po_nrl.pytorch_api.set_device(conf.use_gpu)
    po_nrl.pytorch_api.set_tensorboard_logging(conf.tensorboard_logdir)

    net = Net(
        input_size=1,
        output_size=2,
        layer_size=conf.network_size,
        dropout_rate=conf.dropout_rate,
    ).to(po_nrl.pytorch_api.device())

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=conf.learning_rate
    )

    criterion = torch.nn.CrossEntropyLoss()

    for episode in range(conf.episodes):

        net.train()
        logits = net(torch.tensor([random.randint(0, 1)]).float())  # input 0 or 1
        target = torch.tensor([int(random.random() < .85)])  # output 1 with 85%

        loss = criterion(logits.unsqueeze(0), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        for i in [0, 1]:
            with torch.no_grad():
                prob_distr = np.array([
                    torch.distributions.utils.logits_to_probs(
                        net(torch.tensor([i]).float())
                    )[1]
                    for _ in range(100)
                ])

            po_nrl.pytorch_api.log_tensorboard(f'prob {i}', prob_distr, episode)

        sys.stdout.write('#')
        sys.stdout.flush()


def parse_arguments() -> Namespace:
    """ parse the script arguments

    Args:

    RETURNS (`Namespace`):

    """

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--episodes",
        default=1000,
        type=int,
        help="number of episodes to run"
    )

    parser.add_argument(
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent"
    )

    parser.add_argument(
        "--network_size",
        help='the number of hidden nodes in the q-network',
        default=32,
        type=int
    )

    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help='enables gpu usage'
    )

    parser.add_argument(
        '--tensorboard_logdir',
        type=str,
        help='the log directory for tensorboard',
        required=True
    )

    parser.add_argument(
        '--perturb_stdev',
        default=0,
        type=float,
        help='the amount of parameter pertubation applies during belief updates'
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        help='dropout rate',
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
