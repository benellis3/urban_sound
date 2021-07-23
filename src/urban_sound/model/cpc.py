from typing import Dict, NamedTuple, Tuple
from functools import reduce
from omegaconf.dictconfig import DictConfig
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import operator

patch_typeguard()


class CNNEncoder(nn.Module):
    def __init__(self, size, in_channels, device="cpu"):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.layers = 5
        self.device = device
        self.params = {
            "kernel_size": [10, 8, 4, 4, 4],
            "stride": [5, 4, 2, 2, 2],
            "padding": [3, 2, 1, 1, 1],
        }
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
                )
            )
            self.module_list.append(nn.BatchNorm1d(self.size))

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


ENCODERS = {"cnn": CNNEncoder}


def get_encoder(key):
    return ENCODERS[key]


class AutoRegressiveEncoder(nn.Module):
    def __init__(self, input_size, size, device="cpu"):
        super().__init__()
        self.size = size
        self.input_size = input_size
        self.device = device
        self.network = nn.GRU(input_size, size, batch_first=True)

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


ARENCODERS = {"gru": AutoRegressiveEncoder}


def get_arencoder(key):
    return ARENCODERS[key]


class RandomNegativeSampleSelector:
    def __init__(self, config: DictConfig):
        self.args = config
        self.N = config.N
        self.look_ahead = config.look_ahead

    @typechecked
    def __call__(
        self,
        batch: TensorType["batch", "z_size", "T"],
    ) -> TensorType["batch", "look_ahead", "N-1", "z_size"]:
        # generate random integers
        idx_0 = th.randint(
            batch.size(0), size=(batch.size(0), self.look_ahead, self.N - 1)
        )
        idx_1 = th.randint(
            batch.size(2), size=(batch.size(0), self.look_ahead, self.N - 1)
        )
        return batch[idx_0, :, idx_1]


NEGATIVE_SELECTORS = {"random": RandomNegativeSampleSelector}


def get_negative_selector(key):
    return NEGATIVE_SELECTORS[key]


Scores = NamedTuple("Scores", [("pos", TensorType), ("neg", TensorType)])


class CPC(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.z_size = config.z_size
        self.c_size = config.c_size
        self.encoder = get_encoder(config.encoder)(self.z_size, device=self.device)
        self.auto_regressive_encoder = get_arencoder(config.arencoder)(
            self.z_size, self.c_size, device=self.device
        )
        self.look_ahead = config.look_ahead
        self.w_ks = nn.ModuleList(
            [
                nn.Linear(self.c_size, self.z_size, bias=False)
                for _ in range(self.look_ahead)
            ]
        )
        self.N = config.N
        self.negative_sample_selector = get_negative_selector(
            config.negative_sample_selector
        )(config)

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
        seq_len = batch.size(-1)
        t = th.randint(
            high=int(seq_len / self.encoder.downsample_factor) - self.look_ahead,
            size=(1,),
        )
        # need to multiply by the downsample_factor because t has been selected
        # in the z-time space. In general have to map between z-time and batch-time
        # (i.e. invert encoder's shape transformation).
        z_t: TensorType["batch", "t + look_ahead + 1", "z_size"] = self.encoder(
            batch[:, :, : (t + self.look_ahead + 1) * self.encoder.downsample_factor]
        )

        # pos are the examples that we are trying to predict
        pos: TensorType["batch", "look_ahead", "z_size"] = z_t[:, :, t + 1 :].transpose(
            1, 2
        )
        # randomly (or otherwise) drawn samples from the batch.
        neg: TensorType[
            "batch", "look_ahead", "N-1", "z_size"
        ] = self.negative_sample_selector(z_t)
        out, _ = self.auto_regressive_encoder(z_t[:, :, :t])
        # only need to retain the most recent output
        c_t: TensorType["batch", "c_size"] = out[:, -1, :]

        pos_scores = th.empty((batch.size(0), self.look_ahead, 1))
        neg_scores = th.empty((batch.size(0), self.look_ahead, self.N - 1))
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
