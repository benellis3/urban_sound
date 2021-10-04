# urban_sound

This repo contains some tools for unsupervised clustering and detecting. 

## Environment Setup

Unfortunately, obspy is a special dependency that has to be installed via conda. This means to install the required dependencies, create a conda env as 
normal and then run 

```
conda config add-channels conda-forge && conda install obspy
python setup.py develop
```
Python version >3.8 should be supported. 

### Docker Container

#### Building 
In order to build the docker container, run 

```
docker build -t urban_sound:my_tag -f docker/Dockerfile
```
from the root directory of the repository.

#### Running
The docker container requires a number of directories to be mounted in before running a script. For example, assuming the container was built as above and you are 
in the root directory of the repo,
```
docker run -v $(pwd)/outputs:/source/outputs -v $(pwd)/data:/source/data -v $(pwd)/multirun:/source/multirun --ipc=host urban_sound:my_tag [YOUR COMMAND HERE]
```
Note that when trying to use the GPU, you will have to use `nvidia-docker` instead.

## Running the clusterer

The settings are managed by Hydra in the config files found in src/urban_sound/config. The main config file for clustering is config.yaml in that directory. There
are some sensible defaults, so running the command

```
python src/urban_sound/main.py 
```

If you wish to override a parameter, for example the dataset used, you can run

```
python src/urban_sound/main.py ++dataset=elephants
```

for example. Additionally, you may wish to load a saved model, and generate only the clustering plots. For that you can run 

```
python src/urban_sound/main.py ++generate_tsne_only=true ++saved_model=/path/to/saved/model
```

## Running the detector

The main file has is located at `src/urban_sound/detector.py`. Similar syntax as above applies when attempting to override the various configuration settings.

## Running the tests

This can be done simply with 

```
pytest tests/
```

## Generating Spectrograms offline
This is an experimental feature that is currently too slow. The idea is that you should be able to precompute spectrograms instead of loading the rumbles and
computing their spectrograms at runtime. This can be run by `python src/urban_sound/preprocess_spectra.py`.




