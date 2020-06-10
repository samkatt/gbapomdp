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

## Development

* static analysis & formatting: Run `make static_analyse` in root and check
  whether the code follows some basic rules
* testing: Run `make test` in root and check that all pass

### TODO

* [ ] Raise errors on illegal program arguments where appropriate
* [ ] Require parameter for some (now slightly) hidden program arguments
* [ ] Make `--train_offline` parameter a flag
* [ ] Revisit static analysis
