from omegaconf.dictconfig import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.tensorboard.writer import SummaryWriter
from urban_sound.logging.log import get_summary_writer, log_tsne
from urban_sound.model.cpc import CPC, Scores
from torch.utils.data import DataLoader
from torch import optim
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from torchtyping import TensorType
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)


def make_optimiser(model, config):
    optimiser_class = getattr(optim, config.optimiser)
    return optimiser_class(model.parameters(), lr=config.lr)


def _compute_accuracy(scores):
    accuracy = scores.pos > scores.neg
    accuracy = th.all(accuracy, -1)
    return th.mean(accuracy.float())


def compute_cpc_loss(log_scores: Scores):
    scores = th.cat([log_scores.pos, log_scores.neg], dim=-1)
    return -th.mean(F.log_softmax(scores, dim=-1)[:, :, 0])


class Runner:
    def __init__(
        self,
        model: CPC,
        dataloader: DataLoader,
        optimiser: optim.Optimizer,
        config: DictConfig,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimiser = optimiser
        self.config = config
        self.t = 0

    def train(self) -> None:
        # iterate through the dataloader
        self.model.train()
        iter = tqdm(self.dataloader, dynamic_ncols=True)
        for (batch, _) in iter:
            # compute the scores
            log_scores: Scores = self.model(batch)
            loss = compute_cpc_loss(log_scores)
            # compute the loss function
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.t += 1

            if self.config.log_output and self.t % self.config.update_interval == 0:
                loss = loss.item()
                accuracy = _compute_accuracy(log_scores)
                iter.set_postfix({"loss": loss, "accuracy": accuracy})
                summary_writer = get_summary_writer()
                summary_writer.add_scalar("loss/train", loss, global_step=self.t)
                summary_writer.add_scalar(
                    "accuracy/train", accuracy, global_step=self.t
                )
            if self.config.log_output and self.t % self.config.tsne_interval == 0:
                th.save(self.model.state_dict(), Path(os.getcwd()) / f"model_{self.t}")
                embeddings, all_labels = self._generate_tsne_embeddings()
                if self.dataloader.dataset.is_labelled:
                    log_tsne(embeddings, self.config, self.t, labels=all_labels)
                else:
                    log_tsne(embeddings, self.config, self.t)
                self.model.train()

    def _generate_tsne_embeddings(self) -> TensorType["N", "z_size"]:
        self.model.eval()
        embeddings = []
        labels = []
        with th.no_grad():
            for (batch, label) in tqdm(self.dataloader):
                embedding = self.model.generate_embeddings(batch)
                embeddings.append(embedding)
                labels.append(label)
            return (
                th.cat(embeddings).cpu().numpy(),
                th.cat(labels).unsqueeze(1).cpu().numpy(),
            )
