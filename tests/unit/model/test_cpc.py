from torch.functional import norm
from torch.utils.data import Dataset
import torch as th
from typing import Tuple
from torch.utils.data.dataloader import DataLoader
from torchtyping import TensorType
import pytest
from urban_sound.model.cpc import (
    AutoRegressiveEncoder,
    CNNEncoder,
    CPC,
    RandomNegativeSampleSelector,
)
from unittest.mock import Mock, patch


class SortedNumbersDataset(Dataset):
    """A dataset consisting of different length sequences of sorted integers"""

    def __init__(self, N: int, max_seq_len: int):
        """

        Args:
            N (int): number of points in the dataset to generate
            max_seq_len (int): maximum length of the sequence
        """
        self.max_seq_len = max_seq_len
        self.N = N
        self.data = self._create_data()

    def _create_data(self):
        start_numbers = th.randint(self.N, size=(self.N,))
        seq_len = th.randint(self.max_seq_len, size=(self.N,))
        out = th.zeros(self.N, self.max_seq_len)
        for k in range(self.N):
            out[k, : seq_len[k]] = th.arange(
                start_numbers[k], start_numbers[k] + seq_len[k]
            )
        return out

    def __len__(self):
        return self.N

    def __getitem__(self, index) -> Tuple[TensorType, int]:
        return self.data[index]


@pytest.fixture
def sorted_int_data():
    data = SortedNumbersDataset(256, 128)
    return DataLoader(data, batch_size=8)


@pytest.fixture
def normal_data():
    initialise_rngs(0)
    batch = 10
    channels = 3
    seq_len = 1600
    return th.randn((batch, channels, seq_len))


def initialise_rngs(val):
    th.manual_seed(val)


def test_cnn_encoder_shape(normal_data):
    z_size = 64
    batch_size = normal_data.size(0)
    channels = normal_data.size(1)
    seq_len = normal_data.size(2)
    encoder = CNNEncoder(z_size, channels)
    output = encoder(normal_data)
    assert output.shape == (
        batch_size,
        z_size,
        int(seq_len / encoder.downsample_factor),
    )


def test_encoder_downsample():
    encoder = CNNEncoder(10, 5)
    assert encoder.downsample_factor == 160


def test_cnn_decoder_shape(normal_data):
    output_size = 64
    arencoder = AutoRegressiveEncoder(normal_data.size(1), output_size)

    out, hidden = arencoder(normal_data)
    assert out.shape == (normal_data.size(0), normal_data.size(2), output_size)
    assert hidden.shape == (1, normal_data.size(0), output_size)


def test_negative_sample_selector():
    batch_size = 5
    z_size = 8
    T = 6
    N = 4
    look_ahead = 3
    data = th.arange(batch_size * z_size * T)
    data = data.reshape(batch_size, z_size, T)
    config = Mock(look_ahead=look_ahead, N=N)
    negative_sample_selector = RandomNegativeSampleSelector(config)
    idx_0 = th.zeros((batch_size, look_ahead, N - 1), dtype=int)
    idx_1 = th.zeros((batch_size, look_ahead, N - 1), dtype=int)
    with patch("urban_sound.model.cpc.th.randint", side_effect=[idx_0, idx_1]):
        out = negative_sample_selector(data)
        assert out.shape == (batch_size, look_ahead, N - 1, z_size)
        vals = th.tensor([i * T for i in range(z_size)])
        assert th.all(out == vals)


def test_cpc_shape(normal_data):
    # mock the registries
    batch_size = normal_data.size(0)
    z_size = 64
    c_size = 32
    N = 8
    look_ahead = 6
    downsample_factor = 160

    def _encoder(x):
        # mock function to manipulate input shape correctly
        return th.randn((x.size(0), z_size, int(x.size(2) / downsample_factor)))

    def _arencoder(x):
        out = th.randn((x.size(0), x.size(2), c_size))
        hidden = th.randn((1, x.size(0), c_size))
        return out, hidden

    encoder = Mock(side_effect=_encoder, downsample_factor=downsample_factor)
    arencoder = Mock(side_effect=_arencoder)
    config = Mock(
        device="cpu",
        z_size=z_size,
        c_size=c_size,
        N=N,
        look_ahead=look_ahead,
        negative_sample_selector="random",
    )

    # patch the encoder and decoder
    with patch(
        "urban_sound.model.cpc.get_encoder", return_value=Mock(return_value=encoder)
    ), patch(
        "urban_sound.model.cpc.get_arencoder",
        return_value=Mock(return_value=arencoder),
    ):
        cpc = CPC(config)
        out = cpc(normal_data)
        assert hasattr(out, "pos")
        assert hasattr(out, "neg")
        assert out.pos.shape == (batch_size, look_ahead, 1)
        assert out.neg.shape == (batch_size, look_ahead, N - 1)
