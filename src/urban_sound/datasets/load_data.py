from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, Any
from functools import cached_property

from torch.utils.data import Dataset
from pandas import read_csv, concat
from torchaudio import info, load
from torchaudio.sox_effects import apply_effects_tensor
from torchtyping import TensorType, patch_typeguard
import torch as th
import numpy as np
from typeguard import typechecked

from urban_sound.datasets.line_sweep import remove_overlap

patch_typeguard()


def _maybe_make_path(path: Union[Path, str]) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path


def _get_root_dir():
    return Path(__file__).parent.parent.parent.parent


class NumbersDataset(ABC):
    def __len__(self) -> int:
        return self.N

    @typechecked
    def __getitem__(
        self, index: int
    ) -> Tuple[TensorType, Union[TensorType, np.integer]]:
        return self.data[index], self.labels[index]


class ClusteredDataset(NumbersDataset, Dataset):
    def __init__(self, config):
        self.N = config.dataset.N
        self.max_seq_len = config.dataset.max_seq_len
        self.noise = config.dataset.noise
        self.data, self.labels = self._create_data()
        self.is_labelled = config.dataset.is_labelled

    def _create_data(self):
        half_N = int(self.N / 2)
        ones = th.ones((half_N, 1, self.max_seq_len))
        one_labels = th.ones((half_N,))
        zeros = th.zeros((self.N - half_N, 1, self.max_seq_len))
        zero_labels = th.zeros((self.N - half_N))
        ones += th.normal(mean=0.0, std=self.noise, size=ones.shape)
        zeros += th.normal(mean=0.0, std=self.noise, size=zeros.shape)
        return (th.cat([ones, zeros]), th.cat([one_labels, zero_labels]))


class SortedNumbersDataset(NumbersDataset, Dataset):
    """A dataset consisting of different length sequences of sorted integers"""

    def __init__(self, config):
        """

        Args:
            N (int): number of points in the dataset to generate
            max_seq_len (int): maximum length of the sequence
        """
        self.max_seq_len = config.dataset.max_seq_len
        self.N = config.dataset.N
        self.data = self._create_data()
        self.labels = th.zeros((self.N,))
        self.is_labelled = config.dataset.is_labelled

    def _create_data(self):
        start_numbers = th.randint(self.N, size=(self.N,))
        seq_len = th.randint(self.max_seq_len, size=(self.N,))
        out = th.zeros(self.N, 1, self.max_seq_len)
        for k in range(self.N):
            # need to unsqueeze because the expected data format is
            # (channels, length) and we need 1 channel
            out[k, :, : seq_len[k]] = th.arange(
                start_numbers[k], start_numbers[k] + seq_len[k]
            ).unsqueeze(0)
        return out


class AudioDataset(ABC):
    """
    Expects there to be one metadata object for the whole dataset, and there to be
    a 'start' and 'end' column that give the start and end times in seconds of the sample.
    """

    @abstractmethod
    def _get_audio_path(self, index: int):
        """
        A method to get the path of the audio file from
        an index into the metadata object
        """
        pass

    def _get_metadata_item(self, index: int, column: int):
        return self.metadata.iloc[index, self.metadata.columns.get_loc(column)]

    def __len__(self):
        return len(self.metadata)

    @cached_property
    def max_seq_length(self):
        length_seconds = (
            self.metadata["end"] - self.metadata["start"]
        ).max() + self.grace_period  # add a few seconds grace -- the annotations are not exact
        sample_rate = self.sample_rate
        length_samples = int(np.ceil(length_seconds * sample_rate))
        return length_samples

    def pad_to_max_length(self, audio_tensor, sample_rate):
        assert audio_tensor.size(1) > 0, "Cannot pad empty tensor"
        padding = self.max_seq_length - audio_tensor.size(1)
        assert padding > -1
        effects = [
            ["pad", "0", f"{padding}s"],
            ["channels", "1"],
        ]
        out, _ = apply_effects_tensor(
            audio_tensor, sample_rate, effects, channels_first=True
        )
        assert not th.all(out == 0.0), "Padded audio should not be all silence"
        return out

    def resample(self, audio, sample_rate):
        effects = [["rate", f"{self.sample_rate}"]]
        out, _ = apply_effects_tensor(audio, sample_rate, effects, channels_first=True)
        assert out.size(1) > 0, "Resampled to an empty tensor"
        return out

    def _load_with_slicing(self, index, audio_path, audio_metadata):
        start_frame = int(
            audio_metadata.sample_rate * self._get_metadata_item(index, "start")
        )
        end_frame = int(
            np.ceil(audio_metadata.sample_rate * self._get_metadata_item(index, "end"))
        )
        # load only the slice between start and end
        audio, sample_rate = load(
            filepath=audio_path,
            frame_offset=start_frame,
            num_frames=(end_frame - start_frame + 1),
        )
        return audio, sample_rate

    @typechecked
    def __getitem__(self, index: int) -> Tuple[Any, np.integer]:
        # get the info of the audio
        audio_path = self._get_audio_path(index)
        audio_metadata = info(audio_path)
        if not self.is_pre_sliced:
            audio, sample_rate = self._load_with_slicing(
                index, audio_path, audio_metadata
            )
        else:
            audio, sample_rate = load(filepath=audio_path)
        assert audio.size(1) > 0 and not th.all(
            audio == 0.0
        ), "Loaded a null audio file"
        audio = self.resample(audio, sample_rate)
        audio = self.pad_to_max_length(audio, sample_rate)
        if self.transform:
            audio = self.transform(audio)
        label = self._get_metadata_item(index, "classID")
        if self.label_transform:
            label = self.label_transform(label)
        return audio, label


class BirdDataset(AudioDataset, Dataset):
    """
    Dataset object for the Eastern North American Bird dataset.
    Flattens all the text files into one metadata object.
    """

    def __init__(self, config, transform=None, label_transform=None):
        root_dir = _get_root_dir()
        self.audio_dir = _maybe_make_path(root_dir / config.dataset.audio_dir)
        self.metadata = self._construct_metadata(
            _maybe_make_path(root_dir / config.dataset.metadata_dir)
        )
        self.transform = transform
        self.label_transform = label_transform
        self.is_labelled = config.dataset.is_labelled
        self.is_pre_sliced = config.dataset.is_pre_sliced
        self.sample_rate = config.dataset.sample_rate
        self.grace_period = config.dataset.grace_period

    def _clean_polyphony(self) -> None:
        """Remove all instances of polyphony (multiple sounds co-occurring) from the
        dataset
        """
        for file_name in self.metadata["file_name"].unique():
            metadata = self.metadata[self.metadata["file_name"] == file_name]
            starts = metadata["start"].values
            ends = metadata["end"].values
            indices = metadata.index
            indices_to_delete = remove_overlap(indices, starts, ends)
            self.metadata.drop(indices_to_delete, inplace=True)

    def _get_audio_path(self, index):
        """Function to go from a metadata file path to the audio recording path"""
        metadata_file = self._get_metadata_item(index, "file_name")
        assert isinstance(metadata_file, Path)
        recording = metadata_file.parent.name
        # need to preserve it as a path to remove suffixes
        name = metadata_file
        while name.suffix:
            name = name.with_suffix("")
        # take the name to avoid overwriting the audio_dir
        name = name.with_suffix(".mp3").name
        return self.audio_dir / recording / name

    def _construct_metadata(self, metadata_dir: Path):
        species = 0
        self.species_to_class_id = {}
        metadata_dfs = []

        def _get_classid_from_species(row):
            # required to modify species from the outer scope
            nonlocal species
            if row["Species"] not in self.species_to_class_id:
                self.species_to_class_id[row["Species"]] = species
                species = species + 1
            return self.species_to_class_id[row["Species"]]

        for metadata_file in metadata_dir.rglob("*.txt"):
            metadata = read_csv(metadata_file, sep="\s+")
            if metadata.empty:
                continue
            # add column for file name
            metadata["file_name"] = metadata_file.absolute()
            # rename the columns to the expected format
            metadata = metadata.rename(
                {
                    "begin_time": "start",
                    "end_time": "end",
                },
                axis="columns",
            )

            metadata["classID"] = metadata.apply(_get_classid_from_species, axis=1)
            metadata_dfs.append(metadata)
        return concat(metadata_dfs)


class Urban8KDataset(AudioDataset, Dataset):
    """
    Dataset object for the Urban8k dataset.
    """

    # This ignores the fold-structure of the dataset because we will be doing
    # unsupervised learning with it and hence cross-validation doesn't necessarily make sense.
    def __init__(self, config, transform=None, label_transform=None):
        root_dir = _get_root_dir()
        self.metadata = read_csv(root_dir / config.dataset.metadata_file)
        self.dir = Path(root_dir / config.dataset.audio_dir)
        self.transform = transform
        self.label_transform = label_transform
        self.is_labelled = config.dataset.is_labelled
        self.is_pre_sliced = config.dataset.is_pre_sliced
        self.sample_rate = config.dataset.sample_rate
        self.grace_period = config.dataset.grace_period

    def _get_audio_path(self, index: int) -> Path:
        fold = f"fold{self._get_metadata_item(index, 'fold')}"
        return self.dir / fold / self._get_metadata_item(index, "slice_file_name")
