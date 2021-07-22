from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import Dataset
from pandas import read_csv, concat
from torchaudio import info, load
from torchtyping import TensorType
import torch as th
import numpy as np


def _maybe_make_path(path: Union[Path, str]) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path


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

    def _get_metadata_item(self, index, column):
        return self.metadata.iloc[index, self.metadata.columns.get_loc(column)]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # get the info of the audio
        audio_path = self._get_audio_path(index)
        audio_metadata = info(audio_path)
        start_frame = int(
            audio_metadata.sample_rate * self._get_metadata_item(index, "start")
        )
        end_frame = int(
            np.ceil(audio_metadata.sample_rate * self._get_metadata_item(index, "end"))
        )
        # load only the slice between start and end
        audio = load(
            filepath=audio_path,
            frame_offset=start_frame,
            num_frames=(end_frame - start_frame + 1),
        )
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

    def __init__(self, metadata_dir, audio_dir, transform=None, label_transform=None):
        self.audio_dir = _maybe_make_path(audio_dir)
        self.metadata = self._construct_metadata(_maybe_make_path(metadata_dir))
        self.transform = transform
        self.label_transform = label_transform

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
    def __init__(self, metadata_file, audio_dir, transform=None, label_transform=None):
        self.metadata = read_csv(metadata_file)
        self.dir = Path(audio_dir)
        self.transform = transform
        self.label_transform = label_transform

    def _get_audio_path(self, index: int):
        fold = f"fold{self._get_metadata_item(index, 'fold')}"
        return self.dir / fold / self._get_metadata_item(index, "slice_file_name")
