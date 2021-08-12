import pathlib
from unittest.mock import Mock, patch
from urban_sound.datasets.load_data import BirdDataset, Urban8KDataset
from pandas import DataFrame


def get_dataset_path(key):
    dataset_path = pathlib.Path(__file__).parent.parent / "resources" / key
    if key == "Urban8K":
        audio_path = dataset_path / "audio"
        metadata_path = dataset_path / "metadata" / "UrbanSound8K.csv"
    elif key == "bird_data":
        audio_path = dataset_path / "Recordings"
        metadata_path = dataset_path / "Annotation_Files"
    return audio_path, metadata_path


def test_urban8k_dataset():
    audio_path, metadata_path = get_dataset_path("Urban8K")
    sample_rate = 16000
    max_length = 4  # must be in sync with the data file
    config = Mock(
        dataset=Mock(
            audio_dir=audio_path,
            metadata_file=metadata_path,
            is_labelled=True,
            sample_rate=sample_rate,
        )
    )
    dataset = Urban8KDataset(config)
    aud, label = dataset[0]
    assert label == 3
    assert aud.shape == (1, sample_rate * max_length)


def test_transform():
    audio_path, metadata_path = get_dataset_path("Urban8K")
    config = Mock(
        dataset=Mock(
            audio_dir=audio_path,
            metadata_file=metadata_path,
            is_labelled=True,
            sample_rate=16000,
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
        )
    )
    dataset = BirdDataset(config)
    _, label = dataset[0]
    assert label == 0
    assert dataset[1][1] == 1


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
