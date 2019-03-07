# Partially Observable Neural Reinforcement Learning

# Use

## Installation
Install the required python packages and dependencies

### Known dependencies

Open AI gym:
`pip install gym`

OpenCV python
`pip instal opencv-python`

ffmpeg encoding
`sudo apt install ffmpeg`

## Run the program
``` main.py ``` is located in ``` src ```

```console
python main.py -D cartpole -v --network_size med
```


# Development

## Documentation

Run ``` doxygen .doxygen.config ``` in root and find documentation in ``` doc
``` folder

## TODO
* update documentation
* come up with good way of styling / indenting the code base
* cleanup code
* add tests

## conventions

* use ```python FIXME ``` to identify fixes
* use ```python TODO ``` to identify TODOS
* use Pydoc style comments / docstrings

## arbitrary decisions

* usage of huber loss (over RMSE, unless specified otherwise with --loss rmse)
* usage of Glorot normal initialization for all layers

