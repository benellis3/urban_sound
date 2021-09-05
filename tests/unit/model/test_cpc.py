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

MUT = "urban_sound.model.cpc"


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
    encoder = CNNEncoder(z_size, channels, "cnn_tiny")
    output = encoder(normal_data)
    assert output.shape == (
        batch_size,
        z_size,
        int(seq_len / encoder.downsample_factor),
    )


def test_encoder_downsample():
    encoder = CNNEncoder(10, 5, "cnn_large")
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
    config = Mock(model=Mock(look_ahead=look_ahead, N=N))
    negative_sample_selector = RandomNegativeSampleSelector(config)
    idx_0 = th.zeros((batch_size, look_ahead, N - 1), dtype=int)
    idx_1 = [th.zeros((look_ahead, N - 1), dtype=int) for _ in range(batch_size)]
    with patch(f"{MUT}.th.randint", side_effect=[idx_0, *idx_1]) as rand:
        seq_lens = th.arange(start=1, end=batch_size + 1, dtype=int)
        out = negative_sample_selector(data, seq_lens)
        assert out.shape == (batch_size, look_ahead, N - 1, z_size)
        vals = th.tensor([i * T for i in range(z_size)])
        assert th.all(out == vals)
        for i in range(batch_size):
            rand.assert_any_call(seq_lens[i], size=(look_ahead, N - 1))
        rand.assert_any_call(batch_size, size=(batch_size, look_ahead, N - 1))


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
        model=Mock(
            z_size=z_size,
            c_size=c_size,
            N=N,
            look_ahead=look_ahead,
            negative_sample_selector="random",
            look_ahead_layer="linear",
            embedding_generator="mean_c",
        ),
    )

    # patch the encoder and decoder
    with patch(f"{MUT}.get_encoder", return_value=Mock(return_value=encoder)), patch(
        f"{MUT}.get_arencoder",
        return_value=Mock(return_value=arencoder),
    ):
        cpc = CPC(config)
        out = cpc(normal_data)
        assert hasattr(out, "pos")
        assert hasattr(out, "neg")
        assert out.pos.shape == (batch_size, look_ahead, 1)
        assert out.neg.shape == (batch_size, look_ahead, N - 1)


@pytest.mark.parametrize(
    ("z_t", "c_t", "negative_samples", "pos", "neg"),
    [
        (
            th.Tensor([[[20.0, 21.0]], [[22.0, 23.0]], [[24.0, 25.0]]]),
            th.Tensor([[[-1.0]], [[-3.0]], [[-5.0]]]),
            th.Tensor([[[[22.0]]], [[[24.0]]], [[[20.0]]]]),
            th.Tensor([[[-21.0]], [[-69.0]], [[-125.0]]]),
            th.Tensor([[[-22.0]], [[-72.0]], [[-100.0]]]),
        ),
        (
            th.Tensor(
                [
                    [[13.0, 14.0], [15.0, 16.0]],
                    [[17.0, 18.0], [19.0, 20.0]],
                    [[21.0, 22.0], [23.0, 24.0]],
                ]
            ),
            th.Tensor([[[-1.0, -1.0]], [[-2.0, -2.0]], [[-3.0, -3.0]]]),
            th.Tensor([[[[19.0, 20.0]]], [[[23.0, 24.0]]], [[[13.0, 14.0]]]]),
            th.Tensor([[[-30.0]], [[-76.0]], [[-138.0]]]),
            th.Tensor([[[-39.0]], [[-94.0]], [[-81.0]]]),
        ),
    ],
)
def test_cpc(z_t, c_t, negative_samples, pos, neg):
    """Test which mocks the encoder, the context encoder the negative sample selector
    and checks that the score is calculated correctly."""
    z_size = z_t.size(1)
    c_size = c_t.size(1)
    N = 2
    downsample_factor = 2
    look_ahead = 1
    batch = th.Tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0]]]
    )
    encoder = Mock(side_effect=lambda _: z_t, downsample_factor=downsample_factor)
    ar_encoder = Mock(side_effect=lambda _: (c_t, None))
    look_ahead_layer = Mock(side_effect=lambda x: x)

    def neg_sample_func(batch, seq_lens):
        return negative_samples

    negative_selector = Mock(side_effect=neg_sample_func)
    config = Mock(
        device="cpu",
        model=Mock(
            z_size=z_size,
            c_size=c_size,
            N=N,
            look_ahead=look_ahead,
            negative_sample_selector="random",
            look_ahead_layer="linear",
            embedding_generator="mean_c"
        )
    )
    with patch(f"{MUT}.get_encoder", return_value=Mock(return_value=encoder)), patch(
        f"{MUT}.get_arencoder", return_value=Mock(return_value=ar_encoder)
    ), patch(
        f"{MUT}.get_negative_selector",
        return_value=Mock(return_value=negative_selector),
    ), patch(
        f"{MUT}.nn.ModuleList", return_value=[look_ahead_layer]
    ):
        cpc = CPC(config)
        out = cpc(batch)
        assert th.allclose(out.pos, pos)
        assert th.allclose(out.neg, neg)
        # check the negative selector was called correctly


def test_find_seq_lens():
    # check that the randint calls are made correctly
    config = Mock(
        model=Mock(
            z_size=10,
            c_size=10,
            N=5,
            look_ahead=1,
            negative_sample_selector="random",
            look_ahead_layer="linear",
            embedding_generator="mean_c"
        )
    )

    batch = th.tensor(
        [
            [[1.4, 2.3, 3.0, 0.0, 0.0, 0.0, 0.0], [2.1, 1.0, 1.1, 1.5, 0.0, 0.0, 0.0]],
            [[3.2, 1.3, 9.0, 4.5, 6.0, 7.5, 1.2], [1.0, 1.0, 1.0, 1, 1, 1, 1]],
            [[2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )

    with patch(f"{MUT}.get_encoder", return_value=Mock()), patch(
        f"{MUT}.get_arencoder", return_value=Mock()
    ), patch(f"{MUT}.get_negative_selector", return_value=Mock(),), patch(
        f"{MUT}.nn.ModuleList", return_value=[Mock()]
    ), patch(
        f"{MUT}.get_look_ahead_layers", return_value=Mock()
    ):
        cpc = CPC(config)
        seq_lens = cpc._find_seq_lens(batch)
        assert th.all(seq_lens == th.tensor([4, 7, 1]))
