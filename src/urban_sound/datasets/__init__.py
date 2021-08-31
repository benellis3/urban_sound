from pathlib import Path


from urban_sound.datasets.load_data import (
    ClusteredDataset,
    RumbleOnlyElephantData,
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
    "rumble_only_elephant": RumbleOnlyElephantData,
}


def get_dataset(config):
    return DATASETS[config.dataset.name](config)
