from typing import Dict, List, NamedTuple, Tuple
from functools import reduce
from omegaconf.dictconfig import DictConfig
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import operator
import logging

patch_typeguard()
LOG = logging.getLogger(__name__)

ARCHITECTURE_REGISTRY = {
    "cnn_huge": {
        "kernel_size": [10, 10, 8, 4, 4, 4, 4],
        "stride": [5, 5, 4, 2, 2, 2, 2],
        "padding": [3, 3, 2, 1, 1, 1, 1],
    },
    "cnn_large": {
        "kernel_size": [10, 8, 4, 4, 4],
        "stride": [5, 4, 2, 2, 2],
        "padding": [3, 2, 1, 1, 1],
    },
    "cnn_small": {
        "kernel_size": [10, 4, 4, 4, 4],
        "stride": [5, 2, 2, 2, 2],
        "padding": [3, 1, 1, 1, 1],
    },
    "cnn_tiny": {
        "kernel_size": [4, 4, 4, 4],
        "stride": [2, 2, 2, 2],
        "padding": [1, 1, 1, 1],
    },
}


def get_params(architecture_key: str) -> Dict[str, Dict[str, List[int]]]:
    return ARCHITECTURE_REGISTRY[architecture_key]


class CNNEncoder(nn.Module):
    def __init__(self, size, in_channels, architecture_key, device="cpu"):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.device = device
        self.params = get_params(architecture_key=architecture_key)
        self.layers = len(self.params["kernel_size"])
        assert self.layers == len(self.params["kernel_size"])
        assert self.layers == len(self.params["stride"])
        assert self.layers == len(self.params["padding"])
        self.module_list = nn.ModuleList()
        for i in range(self.layers):
            input_size = size if i > 0 else self.in_channels
            self.module_list.append(
                nn.Conv1d(
                    input_size,
                    self.size,
                    kernel_size=self.params["kernel_size"][i],
                    stride=self.params["stride"][i],
                    padding=self.params["padding"][i],
                    bias=False,
                ).to(self.device)
            )
            self.module_list.append(nn.BatchNorm1d(self.size).to(self.device))

    @typechecked
    def forward(
        self, input: TensorType["batch", "channels", "length"]
    ) -> TensorType["batch", -1, -1]:
        x = input.to(self.device)
        for layer in range(self.layers):
            x = self.module_list[layer * 2](x)
            x = self.module_list[layer * 2 + 1](x)
            x = F.relu(x, inplace=True)
        return x

    @property
    def downsample_factor(self):
        return reduce(operator.mul, self.params["stride"])


ENCODERS_REGISTRY = {
    "cnn_small": CNNEncoder,
    "cnn_large": CNNEncoder,
    "cnn_tiny": CNNEncoder,
    "cnn_huge": CNNEncoder,
}


def get_encoder(key):
    return ENCODERS_REGISTRY[key]


class AutoRegressiveEncoder(nn.Module):
    def __init__(self, input_size, size, device="cpu"):
        super().__init__()
        self.size = size
        self.input_size = input_size
        self.device = device
        self.network = nn.GRU(input_size, size, batch_first=True).to(self.device)

    @typechecked
    def forward(
        self, x: TensorType["batch", "items", "seq_len"]
    ) -> Tuple[TensorType["batch", "seq_len", "out"], TensorType[1, "batch", "out"]]:
        """

        Returns
        -------
        output -- the output of the RNN
        hidden -- the final hidden state of the module
        """
        return self.network(th.transpose(x.to(self.device), 1, 2))

    def init_hidden(self, batch_size):
        raise NotImplementedError("No init hidden")


ARENCODERS_REGISTRY = {"gru": AutoRegressiveEncoder}


def get_arencoder(key):
    return ARENCODERS_REGISTRY[key]


class RandomNegativeSampleSelector:
    def __init__(self, config: DictConfig):
        self.args = config
        self.N = config.model.N
        self.look_ahead = config.model.look_ahead

    @typechecked
    def __call__(
        self, batch: TensorType["batch", "z_size", "T"], seq_lens
    ) -> TensorType["batch", "look_ahead", "N-1", "z_size"]:
        # generate random integers.
        idx_0 = th.randint(
            batch.size(0), size=(batch.size(0), self.look_ahead, self.N - 1)
        )
        idx_1 = []
        for i in range(batch.size(0)):
            idx_1.append(th.randint(seq_lens[i], size=(self.look_ahead, self.N - 1)))
        return batch[idx_0, :, th.stack(idx_1)]


NEGATIVE_SELECTORS_REGISTRY = {"random": RandomNegativeSampleSelector}


def get_negative_selector(key):
    return NEGATIVE_SELECTORS_REGISTRY[key]


LOOK_AHEAD_LAYERS_REGISTRY = {"linear": nn.Linear}


def get_look_ahead_layers(key):
    return LOOK_AHEAD_LAYERS_REGISTRY[key]


class MeanCEmbeddingGenerator:
    """Class which generates an embedding from
    the c_t across time by simply averaging them
    """

    def __init__(self, config):
        pass

    def __call__(self, z_t, c_t, seq_lens, label):
        # need to include only the items before seq_lens[i] because the rest is
        # an encoding of zeros.
        the_sum = th.stack(
            [th.sum(c_t[i, : seq_lens[i], :], dim=0) for i in range(c_t.size(0))]
        )
        return the_sum / seq_lens.unsqueeze(1).repeat(1, c_t.size(2)), label


class MeanZEmbeddingGenerator:
    def __init__(self, config):
        pass

    def __call__(self, z_t, c_t, seq_lens, label):
        the_sum = th.stack(
            [th.sum(z_t[i, :, : seq_lens[i]], dim=1) for i in range(z_t.size(0))]
        )
        return the_sum / seq_lens.unsqueeze(1).repeat(1, z_t.size(1)), label


class ZIdentityEmbeddingGenerator:
    def __init__(self, config):
        pass

    def __call__(self, z_t, c_t, seq_lens, labels):
        zs = [
            (z_t[i, :, : seq_lens[i]], labels[i].unsqueeze(0).repeat(seq_lens[i]))
            for i in range(z_t.size(0))
        ]
        zs = [(z.transpose(0, 1), label) for z, label in zs]
        return th.cat([z[0] for z in zs]), th.cat([z[1] for z in zs])


class CIdentityEmbeddingGenerator:
    def __init__(self, config):
        pass

    def __call__(self, z_t, c_t, seq_lens, labels):
        cs = [
            (c_t[i, : seq_lens[i], :], labels[i].repeat(seq_lens[i]))
            for i in range(c_t.size(0))
        ]
        return th.cat([c[0] for c in cs]), th.cat([c[1] for c in cs])


EMBEDDING_GENERATORS_REGISTRY = {
    "mean_c": MeanCEmbeddingGenerator,
    "mean_z": MeanZEmbeddingGenerator,
    "identity_c": CIdentityEmbeddingGenerator,
    "identity_z": ZIdentityEmbeddingGenerator,
}


def get_embedding_generator(key):
    return EMBEDDING_GENERATORS_REGISTRY[key]


Scores = NamedTuple("Scores", [("pos", TensorType), ("neg", TensorType)])


class CPC(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.z_size = config.model.z_size
        self.c_size = config.model.c_size
        self.encoder = get_encoder(config.model.encoder)(
            self.z_size,
            config.dataset.channels,
            config.model.encoder,
            device=self.device,
        )
        self.auto_regressive_encoder = get_arencoder(config.model.arencoder)(
            self.z_size, self.c_size, device=self.device
        )
        self.look_ahead = config.model.look_ahead
        self.w_ks = nn.ModuleList(
            [
                get_look_ahead_layers(config.model.look_ahead_layer)(
                    self.c_size, self.z_size, bias=False
                ).to(self.device)
                for _ in range(self.look_ahead)
            ]
        )
        self.N = config.model.N
        self.negative_sample_selector = get_negative_selector(
            config.model.negative_sample_selector
        )(config)
        self.embedding_generator = get_embedding_generator(
            config.model.embedding_generator
        )(config)

    def generate_embeddings(
        self,
        batch: TensorType["batch", "channels", "time"],
        label: TensorType["batch", "labels"],
    ) -> TensorType["batch", "c_size"]:
        batch = batch.to(self.device)
        z_t = self.encoder(batch)
        c_t, _ = self.auto_regressive_encoder(z_t)
        seq_lens = self._find_seq_lens(batch)
        seq_lens = (
            th.floor(seq_lens / self.encoder.downsample_factor).long()
            - self.look_ahead
            + 1
        )
        seq_lens = th.maximum(seq_lens, th.zeros_like(seq_lens) + 2)
        return self.embedding_generator(z_t, c_t, seq_lens, label)

    def _find_seq_lens(self, batch):
        # work out where the mask is zero
        non_zero_indices = batch != 0.0
        # find the max of the cumulative sum
        _, seq_lens = th.max(non_zero_indices.cumsum(2), dim=2)
        # take the max across the channels, and convert to a length,
        # not an index
        return (seq_lens + 1).max(dim=1).values

    @typechecked
    def forward(
        self, batch: TensorType["batch", "channels", "time"]
    ) -> Tuple[
        TensorType["batch", "look_ahead", 1], TensorType["batch", "look_ahead", "N-1"]
    ]:
        """
        Selects {self.N} examples from the current batch and computes f_k for all the samples
        returns f_k as well as the classification probabilities

        Args:
            batch (torch.Tensor): the batch to process. Has dimension
            (N, C, T) where N is the batch size, C is the number of channels and
            T is the length of the sequence.

        Returns:
            log_f_k (torch.Tensor): The logarithm of the score of all the z_{t+k} with c_t.
        """
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        seq_lens = self._find_seq_lens(batch)
        seq_lens = [
            max(
                int(seq_lens[i] / self.encoder.downsample_factor) - self.look_ahead + 1,
                2,
            )
            for i in range(batch_size)
        ]
        # t represents the first timestamp in z-space that we do NOT use for the context.
        ts = th.empty((batch_size,), dtype=int)
        for i in range(batch_size):
            ts[i] = th.randint(
                low=1,
                high=seq_lens[i],
                size=(1,),
            )
        max_t = th.max(ts).item()
        # need to multiply by the downsample_factor because t has been selected
        # in the z-time space. In general have to map between z-time and batch-time
        # (i.e. invert encoder's shape transformation).
        z_t: TensorType["batch", "z_size", "t + look_ahead"] = self.encoder(batch)

        # pos are the examples that we are trying to predict
        pos: TensorType["batch", "look_ahead", "z_size"] = th.stack(
            [z_t[i, :, ts[i] : ts[i] + self.look_ahead] for i in range(batch_size)]
        ).transpose(1, 2)
        # randomly (or otherwise) drawn samples from the batch.
        neg: TensorType[
            "batch", "look_ahead", "N-1", "z_size"
        ] = self.negative_sample_selector(z_t, seq_lens=seq_lens)
        out, _ = self.auto_regressive_encoder(z_t[:, :, :max_t])
        # only need to retain the most recent output
        c_t: TensorType["batch", "c_size"] = out[range(batch_size), ts - 1, :]

        pos_scores = th.empty((batch_size, self.look_ahead, 1))
        neg_scores = th.empty((batch_size, self.look_ahead, self.N - 1))
        for k in range(self.look_ahead):
            transformed_c_t: TensorType["batch", "z_size"] = self.w_ks[k](c_t)
            # expand the dims to be the same size as neg
            transformed_c_t = transformed_c_t[:, None, :].repeat(1, self.N - 1, 1)
            neg_scores[:, k, :] = th.sum(
                transformed_c_t[:, : self.N - 1, :] * neg[:, k, :, :], dim=-1
            )

            pos_scores[:, k, :] = th.sum(
                transformed_c_t[:, 0, :] * pos[:, k, :], dim=-1
            ).unsqueeze(-1)
        return Scores(pos=pos_scores, neg=neg_scores)
