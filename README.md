# Dota Squad

Dense neural network for dota pick recommendations 

## Installation

First, create a Python virtual environment (recommended),
```
python3 -m virtualenv .env
```
Activate the virtual environment and cd into the desired parent directory. Then
clone the `dotasquad` repository,
```
git clone git@github.com:morelandjs/dotasquad.git
```
Once you've cloned the repository, cd into the repo directory and install using
pip,
```
pip install tensorflow fuzzywuzzy xdg
```

Note, tensorflow can also be installed with GPU support using the package `tensorflow-gpu`.
See the tensorflow [https://www.tensorflow.org/install/install_linux](installation instructions) for additional details.

## Usage

Train the model by calling dotasquad with optional arguments to specify the
number of training steps and mmr range for the training data.
```
python3 dotasquad.py --train-steps 100000 --mmr-range 2500 3500
```
Training the model could take some time. Tensorflow reports both the training step
and cross entropy loss for each batch to monitor progress.

The model is periodically saved during training to the `$XDG_DATA_HOME/dota`. If
the cache files exist when training is started, tensorflow will attempt to
resume training using the weights 
