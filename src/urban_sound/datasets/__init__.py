from pathlib import Path


from urban_sound.datasets.load_data import (
    ClusteredDataset,
    ContinuousRumbleData,
    ContinuousRumbleImageData,
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
    "continuous_elephant": ContinuousRumbleData,
    "continuous_image_elephant": ContinuousRumbleImageData,
}


def get_dataset(config):
    return DATASETS[config.dataset.name](config)
