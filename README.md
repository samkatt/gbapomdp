# Partially Observable Neural Reinforcement Learning

A code base to run (Model-based Bayesian Neural) Reinforcement Learning
experiments on partially observable domains. This project is meant for
reinforcement learning researchers to compare different methods. It contains
various different environments to test the methods on. Note that this project
has mostly been written for personal use, research, and thus may lack the
documentation that one would typically expect from open source projects.

## Use

### Installation
Install the required python packages and dependencies:

```bash
pip install -r requirements.txt
```

ffmpeg encoding
```bash
sudo apt install ffmpeg
```

### Run the program

Main scripts, `pouct_planning.py`, `model_based.py` and `model_free.py` are
located in `scripts/experiments`

```bash
python {script} -D tiger -v 2
python {script} -h
```

## Development

* static analysis & formatting: Run `./static_analyse.sh` in root and check
  whether the code is formatted correctly
* testing: Run `python setup.py test` in root and check that all pass

### TODO

* [ ] raise errors on illegal program arguments where appropriate
* [ ] require parameter for some (now slightly) hidden program arguments
* [ ] make `--train_offline` parameter a flag
* [ ] revisit static analysis
* [ ] profile (numba)

