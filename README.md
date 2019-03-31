# Partially Observable Neural Reinforcement Learning

A code base to run (Model-based Bayesian Neural) Reinforcement Learning
experiments on partially observable domains. This project is meant for
reinforcement learning researchers to compare different methods. It contains
various different environments to test the methods on. Note that this project
has mostly been written for personal use, research, and thus may lack the
documentation that one would typically expect from open source projects.

## Use

### Installation
Install the required python packages and dependencies

#### Known dependencies

Open AI gym:
```bash pip install gym ```

OpenCV python:
```bash pip instal opencv-python ```

ffmpeg encoding
```bash sudo apt install ffmpeg ```

tensorflow
```bash pip install tensorflow ```

### Run the program
```bash  main.py ``` is located in ``` src ```

```bash
python main.py -D cartpole -v --network_size med
python -h
```

## Development

* documentation: Run ```bash ./make_documentation.sh ``` in root and find
  documentation in ```bash doc/pobnrl ``` folder
* formatting: Run ```bash ./check_formatting.sh ``` in root and check whether the
  code is formatted correctly
* testing: Run ```bash ./run_tests.sh ``` in root and check whether all pass

### TODO
* refactoring
    - rename q_functions and the sort.. what are nets + replay buffers?
    - refactor priors away elegantly
    - create a 'batch' class and use that instead of templates...?
* dev
    - implement static analysing mechanism
    - add levels to verbosity (logging)
* test
    - integration (does everything *run*)
