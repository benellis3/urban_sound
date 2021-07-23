import pathlib

from urban_sound.datasets.load_data import BirdDataset, Urban8KDataset


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
    dataset = Urban8KDataset(metadata_path, audio_path)
    aud, label = dataset[0]
    assert label == 3
    assert aud[0].shape == (2, 14004)
    assert aud[1] == 44100


def test_transform():
    audio_path, metadata_path = get_dataset_path("Urban8K")
    dataset = Urban8KDataset(
        metadata_path,
        audio_path,
        transform=lambda x: "pickles",
        label_transform=lambda x: x + 1,
    )
    aud, label = dataset[0]
    assert aud == "pickles"
    assert label == 4


def test_bird_dataset():
    audio_path, metadata_path = get_dataset_path("bird_data")
    dataset = BirdDataset(metadata_path, audio_path)
    aud, label = dataset[0]
    assert label == 0
    assert aud[0].shape == (1, 28137)
    assert aud[1] == 32000
    assert dataset[1][1] == 1
