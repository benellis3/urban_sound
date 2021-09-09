from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, Any
from functools import cached_property, lru_cache, partial
from pandas.core.frame import DataFrame
from math import pi

from torch.utils.data import Dataset
from pandas import read_csv, concat, Timedelta, Timestamp, to_datetime
from torchaudio import info, load
import torchaudio.functional as audio_F
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import Spectrogram
from torchtyping import TensorType, patch_typeguard
from torchvision.transforms import Resize
import torch.nn.functional as F
import torch as th
import numpy as np
from typeguard import typechecked
from urban_sound.datasets.kenya_data_getter import (
    data_getter,
    generate_image_by_start_end,
)
from obspy.core.utcdatetime import UTCDateTime

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
        if index in self.cache:
            return self.cache[index]
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
        self.cache[index] = (audio, label)
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
        self.cache = {}

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
        self.label_map = {
            0: "air_conditioner",
            1: "car_horn",
            2: "children_playing",
            3: "dog_bark",
            4: "drilling",
            5: "engine_idling",
            6: "gun_shot",
            7: "jackhammer",
            8: "siren",
            9: "street_music",
        }
        self.cache = {}

    def _get_audio_path(self, index: int) -> Path:
        fold = f"fold{self._get_metadata_item(index, 'fold')}"
        return self.dir / fold / self._get_metadata_item(index, "slice_file_name")


class ElephantDataBase:
    def __init__(self, config):
        self.config = config
        root_dir = _get_root_dir()
        self.rumble_only_metadata = read_csv(
            root_dir / config.dataset.directory / "metadata.csv", parse_dates=["events"]
        )
        self.rumble_only_metadata = self.rumble_only_metadata.loc[
            self.rumble_only_metadata["rumble"] == 1
        ]
        self.seismic_data_loader = data_getter(
            seismic_path=root_dir / config.dataset.directory
        )
        # hard coded because reading the data alone takes a long time
        # this means we cannot work out the max length in advanace
        self.max_length = (
            config.dataset.sampling_frequency * 2 * config.dataset.timedelta + 1
        )

    def _read_data(self, station, start, end, components):
        streams = self.seismic_data_loader.get_seismic_cached(
            station, start, end, components
        )
        stream = streams[0] + streams[1] + streams[2]
        assert stream.traces
        data = [t.data for t in stream.traces]
        ret = []
        for datum in data:
            if datum.shape[0] != self.max_length:
                new_data = np.zeros((datum.shape[0] + 1))
                new_data[:-1] = datum
                ret.append(new_data)
            else:
                ret.append(datum)
        ret = np.stack(ret)
        return th.tensor(ret)

    def _get_rumble_only_metadata_item(self, index: int, column: int):
        return self.rumble_only_metadata.iloc[
            index, self.rumble_only_metadata.columns.get_loc(column)
        ]


class RumbleOnlyElephantData(ElephantDataBase, Dataset):
    def __init__(self, config, transform=None):
        super(RumbleOnlyElephantData, self).__init__(config)
        self.transform = transform
        self.timedelta = config.dataset.timedelta
        self.is_labelled = config.dataset.is_labelled
        self.single_station_mode = config.dataset.single_station_mode
        self.station = getattr(config.dataset, "station", None)
        if self.single_station_mode:
            self.rumble_only_metadata = self.rumble_only_metadata.loc[
                self.rumble_only_metadata["station"] == self.station
            ]
        self.station_labels = {"EEL11": 0, "ETA00": 1}
        self.label_map = {0: "EEL11", 1: "ETA00"}
        self._construct_data()

    def __len__(self):
        return len(self.rumble_only_metadata)

    def _construct_data(self):
        # read in the rumble data from the csv file
        self.rumble_only_metadata["start"] = self.rumble_only_metadata[
            "events"
        ] - Timedelta(seconds=self.timedelta)
        self.rumble_only_metadata["start"] = self.rumble_only_metadata["start"].apply(
            UTCDateTime
        )
        self.rumble_only_metadata["end"] = self.rumble_only_metadata[
            "events"
        ] + Timedelta(seconds=self.timedelta)
        self.rumble_only_metadata["end"] = self.rumble_only_metadata["end"].apply(
            UTCDateTime
        )

    @lru_cache(maxsize=1520)
    def __getitem__(self, index):
        start = self._get_rumble_only_metadata_item(index, "start")
        end = self._get_rumble_only_metadata_item(index, "end")
        station = self._get_rumble_only_metadata_item(index, "station")
        components = ["_e_", "_n_", "_z_"]
        data = self._read_data(station, start, end, components)
        return data, th.tensor(self.station_labels[station])


class ContinuousRumbleData(ElephantDataBase, Dataset):
    # Need a consistent ordering over the times
    # ==> can do with continuous time ordering within a station and
    # ==> arbitrary fixed ordering between stations.
    # Up front should create a dataframe with columns:
    # index | file_path | start_time | end_time | rumble
    # so that we can then read out of the dataframe
    def __init__(self, config, transform=None):
        super(ContinuousRumbleData, self).__init__(config)
        self.config = config
        self.transform = transform
        self.path = _get_root_dir() / self.config.dataset.directory
        self.timedelta = self.config.dataset.timedelta
        self.grace_period = self.config.dataset.grace_period
        self.stations = self.config.dataset.stations
        self._construct_metadata()

    def _interpolate_time(self, start: Timestamp, end: Timestamp):
        """interpolates between two times using self.timedelta"""
        end_time = end - Timedelta(seconds=self.grace_period)
        start_time = start + Timedelta(seconds=self.grace_period)
        # have to convert the step to nanoseconds because the values are
        # unix timestamps
        window = np.arange(
            start=start_time.value, stop=end_time.value, step=2 * self.timedelta * 1e9
        )
        df = to_datetime(window)
        df = DataFrame(df, columns=["start"])
        df["rumble"] = 0
        return df

    def _find_rumbles(self, df, event):
        larger_times = df["start"] > event.asm8
        larger_index = (larger_times == 1).idxmax()
        if larger_index == 0:
            # event has not found a match in this time window
            return
        df["rumble"].iloc[larger_index - 1] = 1

    def _construct_metadata_for_station(self, station):
        # load the file times csv
        station_dir = self.path / station
        file_times = read_csv(
            station_dir / "files_times.csv", parse_dates=["start", "end"]
        )
        start_time = file_times["start"].min()
        end_time = file_times["end"].max()
        # ASSUME that the file_times only has 2 groups in it.
        # This is the case for all the data currently
        border_time = file_times["end"].min()
        first_window = self._interpolate_time(start_time, border_time)
        second_window = self._interpolate_time(border_time, end_time)
        station_metadata = self.rumble_only_metadata.loc[
            self.rumble_only_metadata["station"] == station
        ]
        for _, row in station_metadata.iterrows():
            event = row["events"]
            self._find_rumbles(first_window, event)
            self._find_rumbles(second_window, event)
        # merge the two dataframes
        df = concat([first_window, second_window], ignore_index=True)
        df["station"] = station
        return df

    def _construct_metadata(self):
        dfs = []
        for station in self.stations:
            dfs.append(self._construct_metadata_for_station(station))
        self.metadata = concat(dfs, ignore_index=True)
        self.metadata["end"] = self.metadata["start"] + 2 * Timedelta(
            seconds=self.timedelta
        )
        self.metadata["start"] = self.metadata["start"].apply(UTCDateTime)
        self.metadata["end"] = self.metadata["end"].apply(UTCDateTime)

    def __len__(self):
        return len(self.metadata)

    def _get_metadata_item(self, index, column):
        return self.metadata.iloc[index, self.metadata.columns.get_loc(column)]

    def __getitem__(self, index: int):
        station = self._get_metadata_item(index, "station")
        start = self._get_metadata_item(index, "start")
        end = self._get_metadata_item(index, "end")
        components = ["_e_", "_n_", "_z_"]
        label = self._get_metadata_item(index, "rumble")
        return (
            self._read_data(station, start, end, components),
            label,
        )


def tukey_window(alpha, n_fft):
    ret = th.zeros(n_fft)
    half_alpha_n = int((alpha * n_fft / 2))
    ret[:half_alpha_n] = 0.5 * (1 - th.cos(2 * th.tensor(pi) / (alpha * n_fft)))
    ret[half_alpha_n : n_fft // 2 + 1] = 1
    ret[n_fft // 2 + 1 :] = ret[: n_fft // 2]


class ContinuousRumbleImageData(ContinuousRumbleData):
    def __init__(self, config, transform=None):
        super(ContinuousRumbleImageData, self).__init__(config, transform)
        self.resize = self.config.dataset.resize
        self.fmin = self.config.dataset.frequency_min
        self.fmax = self.config.dataset.frequency_max
        self.device = self.config.device
        self.spectrogram = Spectrogram(
            n_fft=self.config.dataset.n_fft,
            win_length=self.config.dataset.win_length,
            hop_length=self.config.dataset.hop_length,
            normalized=False,
            power=1,
            window_fn=partial(tukey_window, 0.25),
        )
        self.resize_layer = Resize((self.config.dataset.size, self.config.dataset.size))

    def __getitem__(self, index: int):
        station = self._get_metadata_item(index, "station")
        start = self._get_metadata_item(index, "start")
        end = self._get_metadata_item(index, "end")
        label = self._get_metadata_item(index, "rumble")
        if self.config.dataset.kenya_data:
            img = th.tensor(
                generate_image_by_start_end(
                    self.seismic_data_loader, station, start, end, resize=self.resize
                )
            )
        elif self.config.dataset.use_collate_fn:
            waveform = self._read_data(station, start, end, ["_e_", "_n_", "_z_"])
            return waveform, label
        else:
            img = self.generate_image(station, start, end, do_resize=self.resize)
        img = img.unsqueeze(0).repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        return img, label

    def generate_image(self, station, start, end, do_resize=False):
        waveform = self._read_data(station, start, end, ["_e_", "_n_", "_z_"])
        waveform = audio_F.highpass_biquad(
            waveform, self.config.dataset.sampling_frequency, self.fmin
        )
        waveform = audio_F.lowpass_biquad(
            waveform, self.config.dataset.sampling_frequency, self.fmax
        )

        waveform = waveform - waveform.mean(dim=1).unsqueeze(1).expand(
            -1, waveform.size(1)
        )
        out = self.spectrogram(waveform)
        img = 10 * th.log10(out[0] + out[1] + out[2])
        img = img[int(self.fmin * 2) : int(self.fmax * 2), :]
        if do_resize:
            img = self.resize_layer(img.unsqueeze(0))
            img = img.squeeze()
        return img


class ElephantImageCollateFunction:
    """A Callable class that creates a spectrogram on a batch
    of seismic audio and, scales the individual components and band-pass
    filters the output. Can be passed as the `collate_fn` param of the
    DataLoader pytorch class in conjunction with the ContinuousRumbleImageData
    dataset above."""

    def __init__(self, config):
        self.config = config
        self.fmin = self.config.dataset.frequency_min
        self.fmax = self.config.dataset.frequency_max
        self.device = self.config.device
        self.spectrogram = Spectrogram(
            n_fft=self.config.dataset.n_fft,
            win_length=self.config.dataset.win_length,
            hop_length=self.config.dataset.hop_length,
            normalized=False,
            power=1,
            window_fn=partial(tukey_window, 0.25),
        )
        self.resize = self.config.dataset.resize

    def __call__(self, batch):
        labels = [item[1] for item in batch]
        labels = th.LongTensor(labels)
        data = th.stack([item[0] for item in batch])
        data = self._generate_image(data)
        data = data.expand(-1, 3, -1, -1)
        return [data, labels]

    def _generate_image(self, data):
        waveform = audio_F.highpass_biquad(
            data, self.config.dataset.sampling_frequency, self.fmin
        )
        waveform = audio_F.lowpass_biquad(
            waveform, self.config.dataset.sampling_frequency, self.fmax
        )

        waveform = waveform - waveform.mean(dim=2).unsqueeze(2).expand(
            -1, -1, waveform.size(2)
        )
        spectrogram = self.spectrogram(waveform)
        spectrogram = 10 * th.log10(spectrogram.sum(dim=1))
        spectrogram = spectrogram[:, int(self.fmin * 2) : int(self.fmax * 2), :]
        spectrogram = spectrogram.unsqueeze(1)
        if self.resize:
            spectrogram = F.interpolate(
                spectrogram, size=(self.config.dataset.size, self.config.dataset.size)
            )
        return spectrogram


class EmbeddingsDataset(Dataset):
    """DataLoader used fetch data from an input tensor"""

    def __init__(
        self, data: TensorType["batch", "size"], labels: TensorType["batch"] = None
    ):
        self.data = th.tensor(data).detach()
        self.labels = th.tensor(labels).detach()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
