import pathlib
from unittest.mock import Mock, patch
from urban_sound.datasets.load_data import (
    BirdDataset,
    ContinuousRumbleData,
    ContinuousRumbleImageData,
    EmbeddingsDataset,
    RumbleOnlyElephantData,
    Urban8KDataset,
)
from urban_sound.datasets import get_dataset
from pandas import DataFrame
import torch as th


def get_dataset_path(key):
    dataset_path = pathlib.Path(__file__).parent.parent / "resources" / key
    if key == "Urban8K":
        audio_path = dataset_path / "audio"
        metadata_path = dataset_path / "metadata" / "UrbanSound8K.csv"
    elif key == "bird_data":
        audio_path = dataset_path / "Recordings"
        metadata_path = dataset_path / "Annotation_Files"
    elif key == "elephants":
        audio_path = dataset_path
        metadata_path = dataset_path / "metadata.csv"
    else:
        raise KeyError(f"{key} not recognised as dataset type")
    return audio_path, metadata_path


def test_urban8k_dataset():
    audio_path, metadata_path = get_dataset_path("Urban8K")
    sample_rate = 16000
    max_length = 4  # must be in sync with the data file
    grace_period = 0.1
    config = Mock(
        dataset=Mock(
            audio_dir=audio_path,
            metadata_file=metadata_path,
            is_labelled=True,
            sample_rate=sample_rate,
            is_pre_sliced=True,
            grace_period=grace_period,
        )
    )
    dataset = Urban8KDataset(config)
    aud, label = dataset[0]
    assert label == 3
    assert aud.shape == (1, sample_rate * (max_length + grace_period))


def test_transform():
    audio_path, metadata_path = get_dataset_path("Urban8K")
    config = Mock(
        dataset=Mock(
            audio_dir=audio_path,
            metadata_file=metadata_path,
            is_labelled=True,
            sample_rate=16000,
            is_pre_sliced=True,
            grace_period=0.1,
        )
    )
    dataset = Urban8KDataset(
        config,
        transform=lambda x: "pickles",
        label_transform=lambda x: x + 1,
    )
    aud, label = dataset[0]
    assert aud == "pickles"
    assert label == 4


def test_bird_dataset():
    audio_path, metadata_path = get_dataset_path("bird_data")
    config = Mock(
        dataset=Mock(
            audio_dir=audio_path,
            metadata_dir=metadata_path,
            is_labelled=True,
            sample_rate=16000,
            is_pre_sliced=False,
            grace_period=0.1,
        )
    )
    dataset = BirdDataset(config)
    _, label = dataset[0]
    assert label == 0
    assert dataset[1][1] == 1


def test_elephant_dataset():
    audio_path, _ = get_dataset_path("elephants")
    config = Mock(
        dataset=Mock(
            timedelta=5,
            directory=audio_path,
            is_labelled=False,
            name="rumble_only_elephant",
            sampling_frequency=200,
            single_station_mode=False,
            station=None,
        )
    )
    dataset = RumbleOnlyElephantData(config)
    data, _ = dataset[0]
    assert data.shape == (3, 2001)
    data, _ = dataset[300]
    assert data.shape == (3, 2001)


def test_continuous_elephant_dataset():
    audio_path, _ = get_dataset_path("elephants")
    config = Mock(
        dataset=Mock(
            timedelta=5,
            directory=audio_path,
            is_labelled=True,
            name="continuous_elephant",
            sampling_frequency=200,
            stations=["EEL11"],
            grace_period=60,
        )
    )
    dataset = ContinuousRumbleData(config)
    data, label = dataset[0]
    assert data.shape == (3, 2001)
    assert label == 0


def test_continuous_image_elephant_dataset():
    audio_path, _ = get_dataset_path("elephants")
    config = Mock(
        dataset=Mock(
            timedelta=5,
            directory=audio_path,
            is_labelled=True,
            name="continuous_image_elephant",
            sampling_frequency=200,
            stations=["EEL11"],
            grace_period=60,
        )
    )
    dataset = ContinuousRumbleImageData(config)
    data, label = dataset[0]
    assert data.shape == (128, 128)
    assert label == 0
    data, label = dataset[51]
    assert data.shape == (128, 128)
    assert label == 1


def test_clean_polyphony():
    df = {
        "start": [0.0, 0.6, 0.3, 0.8, 0.9, 1.5, 2.5],
        "end": [0.7, 0.62, 0.35, 1.0, 1.2, 10.0, 7.0],
        "file_name": ["file1", "file1", "file1", "file1", "file2", "file2", "file2"],
    }
    out = {"start": [0.8, 0.9], "end": [1.0, 1.2], "file_name": ["file1", "file2"]}
    config = Mock(
        dataset=Mock(
            audio_dir="dir",
            metadata_dir="meta_dir",
            sample_rate=16000,
            is_labelled=True,
        )
    )

    with patch.object(BirdDataset, "_construct_metadata", return_value=DataFrame(df)):
        dataset = BirdDataset(config)
        dataset._clean_polyphony()
        assert dataset.metadata.reset_index(drop=True).equals(DataFrame(out))


def test_embeddings_dataset():
    zs = th.zeros((100, 5))
    labels = th.zeros((100,))
    dataset = EmbeddingsDataset(zs, labels)
    z, label = dataset[0]

    assert th.all(z == th.zeros((5,)))
    assert label == 0

    assert len(dataset) == 100
