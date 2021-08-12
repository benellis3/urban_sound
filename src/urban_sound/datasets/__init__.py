from pathlib import Path

from omegaconf.dictconfig import DictConfig

from urban_sound.datasets.load_data import (
    ClusteredDataset,
    Urban8KDataset,
    BirdDataset,
    SortedNumbersDataset,
)

# TODO add config for specifying transforms

DATASETS = {
    "birds": BirdDataset,
    "urban_sound": Urban8KDataset,
    "numbers": SortedNumbersDataset,
    "clusters": ClusteredDataset,
}


def get_dataset(config):
    return DATASETS[config.dataset.name](config)
