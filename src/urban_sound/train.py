from omegaconf.dictconfig import DictConfig
from hydra.utils import to_absolute_path
from torch.profiler.profiler import tensorboard_trace_handler
from torch.utils.tensorboard.writer import SummaryWriter
from urban_sound.logging.log import get_summary_writer, log_tsne
from urban_sound.model.cpc import CPC, Scores
from torch.utils.data import DataLoader
from torch.profiler import profile, schedule, ProfilerActivity
from torch import optim
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from torchtyping import TensorType
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)


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

    def _train_loop(self, iter, profiler=None) -> None:
        for (batch, _) in iter:
            # compute the scores
            log_scores: Scores = self.model(batch)
            loss = compute_cpc_loss(log_scores)
            # compute the loss function
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            if profiler is not None:
                profiler.step()
            self.t += 1

            if (
                self.config.training.log_output
                and self.t % self.config.training.update_interval == 0
            ):
                loss = loss.item()
                accuracy = _compute_accuracy(log_scores)
                iter.set_postfix({"loss": loss, "accuracy": accuracy})
                summary_writer = get_summary_writer()
                summary_writer.add_scalar("loss/train", loss, global_step=self.t)
                summary_writer.add_scalar(
                    "accuracy/train", accuracy, global_step=self.t
                )
            if (
                self.config.training.log_output
                and self.t % self.config.training.tsne_interval == 0
            ):
                fname = (
                    "model_{self.t}.pt"
                    if not hasattr(self.config, "tag")
                    else f"model_{self.config.tag}_{self.t}.pt"
                )
                th.save(
                    self.model.state_dict(),
                    Path(os.getcwd()) / fname,
                )
                self.generate_tsne_embeddings()
                self.model.train()

    def train(self) -> None:
        # iterate through the dataloader
        self.model.train()
        iter = tqdm(self.dataloader, dynamic_ncols=True)
        if self.config.device == "cuda" and self.config.profiler.profile:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(
                schedule=schedule(
                    wait=self.config.profiler.wait,
                    warmup=self.config.profiler.warmup,
                    active=self.config.profiler.active,
                ),
                on_trace_ready=tensorboard_trace_handler,
                activities=activities,
            ) as profiler:
                self._train_loop(iter, profiler=profiler)
        else:
            # profiling does not work on Mac because we don't have Kineto
            # internet does not seem to have a suggestion for how to fix this
            self._train_loop(iter)

    def generate_tsne_embeddings(self, display=False) -> None:
        embeddings, all_labels = self._generate_tsne_embeddings()
        label_map = getattr(self.dataloader.dataset, "label_map", None)
        if self.dataloader.dataset.is_labelled:
            log_tsne(
                embeddings,
                self.config,
                self.t,
                labels=all_labels,
                display=display,
                label_map=label_map,
            )
        else:
            log_tsne(
                embeddings, self.config, self.t, display=display, label_map=label_map
            )

    def _generate_tsne_embeddings(self) -> TensorType["N", "z_size"]:
        if self.config.embeddings.load_embeddings:
            embeddings = th.load(
                self.config.embeddings.embeddings_path, map_location="cpu"
            )
            labels = th.load(self.config.embeddings.labels_path, map_location="cpu")
            return (embeddings.numpy(), labels.numpy())
        self.model.eval()
        embeddings = []
        labels = []
        with th.no_grad():
            for (batch, label) in tqdm(self.dataloader):
                embedding, new_label = self.model.generate_embeddings(batch, label)
                embeddings.append(embedding)
                labels.append(new_label)
            embeddings = th.cat(embeddings).cpu()
            labels = th.cat(labels).unsqueeze(1).cpu().long()
            if self.config.embeddings.save_embeddings:
                th.save(Path(os.getcwd()) / f"embeddings_{self.config.tag}.pt")
                th.save(Path(os.getcwd()) / f"labels_{self.config.tag}.pt")
            return (
                embeddings.numpy(),
                labels.numpy(),
            )
