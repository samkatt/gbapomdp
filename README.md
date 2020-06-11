# Partially Observable Neural Reinforcement Learning

A code base to run (Model-based Bayesian Neural) Reinforcement Learning
experiments on partially observable domains. This project allows reinforcement
learning researchers to compare different methods. It contains environments to
test the methods on. Note that this project is for personal use, research, and
thus may lack the documentation that one would typically expect from open
source projects.

## Use

### Installation

Install the required python packages and dependencies:

```bash
pip install -r requirements.txt
python setup.py install
```

ffmpeg encoding

```bash
sudo apt install ffmpeg
```

### Run the program

Main scripts, `pouct_planning.py`, `model_based.py` are located in
`scripts/experiments`

```bash
python {script} -D tiger -v 2
python {script} -h
```

### Experiments

Some of the experiments required hands-on tinkering of some code. Examples are
freezing the observation model in the collision avoidance experiments.
Additionally, some code tinkering was done after some of the experiments, so
results may change slightly. However, here are some commands ran to reproduce
the results published in BADDr.

####  Tiger

```shell
base_params="-D tiger --runs 10000 --episodes 400 -v 1 -H 30"
planning_params="--expl 100 --num_sims 4096 --num_part 1024 -B importance_sampling --belief_minimal_sample_size 128"
prior_params="--num_pretrain 4096 --alpha .1 --train on_prior --prior_certainty 100000 --num_nets 1"
learn_params="--dropout .5"
extra_params="--prior_correct 0 --online_learning_rate .005 --backprop"

python model_based.py $base_params $planning_params $prior_params $learn_params $extra_params
```

####  Road Race

3 lanes:
```shell
base_params="-D road_racer --domain_size 3 --runs 10000 --episodes 300 -v 1 -H 20"
planning_params="--expl 15 --num_sims 128 --num_part 1024 --search_depth 3 -B rejection_sampling"
prior_params="--num_pretrain 2048 --alpha .005 --train on_prior --prior_cert 1000 --num_nets 1"
learn_params="--network_size 32 --batch_size 64 --online_learning_rate .0001 --backprop"

python model_based.py $base_params $planning_params $prior_params $learn_params $extra_params
```

9 lanes:
```shell

base_params="-D road_racer --domain_size 9 --runs 10000 --episodes 200 -v 1 -H 20"
planning_params="--expl 15 --num_sims 128 --num_part 1024 --search_depth 3 -B rejection_sampling"
prior_params="--num_pretrain 16384 --alpha .0025 --train on_prior --prior_cert 1000 --num_nets 1"
learn_params="--network_size 256 --batch_size 256 --online_learning_rate .0001 --backprop"

python model_based.py $base_params $planning_params $prior_params $learn_params $extra_params
```

## Development

* static analysis & formatting: Run `make static_analyse` in root and check
  whether the code follows some basic rules
* testing: Run `make test` in root and check that all pass
