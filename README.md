# Dota Squad

Dense neural network for DOTA pick recommendations 

## Installation

First, create a Python virtual environment (recommended),
```
python3 -m virtualenv .env
```
Activate the virtual environment and cd into the desired parent directory. Then clone the `dotasquad` repository,
```
git clone git@github.com:morelandjs/dotasquad.git
```
Once you've cloned the repository, cd into the repo directory and install using pip,
```
pip install tensorflow fuzzywuzzy xdg
```

Note, tensorflow can also be installed with GPU support using the package `tensorflow-gpu`. See the tensorflow [installation instructions](https://www.tensorflow.org/install/install_linux) for additional details.

## Usage

Train the model by calling dotasquad with optional arguments to specify the number of training steps and mmr range of the training data.
```
python3 dotasquad.py --train-steps 100000 --mmr-range 2500 3500
```
Training the model could take some time. Tensorflow reports both the training step
and cross entropy loss for each batch of training samples to monitor progress.

The model's progress is periodically saved during training. For example, the network graph and calibrated node weights are cached in `$XDG_DATA_HOME/dota`. If these cache files exist when training is started, tensorflow will attempt to
recover from the last available checkpoint. Consequently, one should delete this
cache directory to retrain the network from a blank slate.

DotaSquad recommends hero picks through the `predict` function. For example,
```
import dotasquad

ally_heroes = ['Lion', 'Witch Doctor']
enemy_heroes = ['Invoker', 'Bristleback', 'Wraith King']

dotasquad.predict(ally_heroes, enemy_heroes)

[('Ally_Anti-Mage', 0.9610115),
 ('Ally_Spectre', 0.021461263),
 ('Ally_Lifestealer', 0.009342404),
 ('Ally_Faceless_Void', 0.006751618),
 ('Ally_Chaos_Knight', 0.00068466447),
 ('Ally_Drow_Ranger', 0.000652589),
...
]
```
returns a list of recommended heroes, sorted from strongest to weakest. Each hero is reported with a corresponding number (probability) that reflects the network's confidence in that pick.
For instance, the previous example overwhelmingly recommends Anti-Mage.
