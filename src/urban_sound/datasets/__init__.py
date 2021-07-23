from pathlib import Path

from urban_sound.datasets.load_data import (
    Urban8KDataset,
    BirdDataset,
    SortedNumbersDataset,
)

# TODO add config for specifying transforms


def _make_bird_dataset(config) -> BirdDataset:
    root_dir = _get_root_dir()
    audio_dir = root_dir / config.audio_dir
    metadata_dir = root_dir / config.metadata_dir
    return BirdDataset(metadata_dir=metadata_dir, audio_dir=audio_dir)


def _make_urban_sound_dataset(config) -> Urban8KDataset:
    root_dir = _get_root_dir()
    audio_dir = root_dir / config.audio_dir
    metadata_file = root_dir / config.metadata_file
    return Urban8KDataset(metadata_file=metadata_file, audio_dir=audio_dir)


def _make_numbers_dataset(config) -> SortedNumbersDataset:
    return SortedNumbersDataset(N=config.N, max_seq_len=config.max_seq_len)


DATASETS = {
    "birds": _make_bird_dataset,
    "urban_sound": _make_urban_sound_dataset,
    "numbers": _make_numbers_dataset,
}

def get_dataset(config):
    return DATASETS[config.dataset](config)

def _get_root_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent
