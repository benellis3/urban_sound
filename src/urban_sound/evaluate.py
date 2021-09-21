import logging
from typing import NamedTuple
import os
from omegaconf import DictConfig
from torch.profiler.profiler import tensorboard_trace_handler
from torch.profiler import profile, schedule, ProfilerActivity
from torch.utils.data.dataloader import DataLoader
from torchtyping import TensorType
from torch import optim
from torch.utils.data import random_split
import torch.nn as nn
import torch as th
from tqdm import tqdm
from urban_sound.datasets.load_data import EmbeddingsDataset
from urban_sound.logging.log import get_summary_writer
from urban_sound.model.eval import SimpleClassifier

LOG = logging.getLogger(__name__)

Results = NamedTuple(
    "Results", [("accuracy", float), ("recall", float), ("precision", float)]
)


class EvalRunner:
    """Simple Runner class to evaluate the quality of embeddings by training
    a classifier
    """

    def __init__(
        self,
        model: SimpleClassifier,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimiser: optim.Optimizer,
        config: DictConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimiser = optimiser
        self.config = config
        self.t = 0
        self.eps = 1e-6

    def _train_loop(self, profiler=None):
        loss = nn.CrossEntropyLoss()
        for (batch, labels) in tqdm(self.train_dataloader):
            self.optimiser.zero_grad()
            predictions = self.model(batch)
            LOG.debug(f"Predictions shape: {predictions.shape}")
            LOG.debug(f"Labels shape: {labels.shape}")
            labels = labels.long().to(self.config.device)
            loss_val = loss(predictions, labels)
            loss_val.backward()
            self.optimiser.step()
            if profiler is not None:
                profiler.step()
            self.t = self.t + 1

    def train(self):
        self.model.train()
        if self.config.device == "cuda" and self.config.profiler.profile:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(
                schedule=schedule(
                    wait=self.config.profiler.wait,
                    warmup=self.config.profiler.warmup,
                    active=self.config.profiler.active,
                ),
                on_trace_ready=tensorboard_trace_handler(os.getcwd()),
                activities=activities,
                with_stack=True,
            ) as profiler:
                self._train_loop(profiler=profiler)
        else:
            self._train_loop()

    def test(self):
        self.model.eval()
        total = 0
        correct = 0
        true_positive = 0
        classifier_positive = 0
        labelled_positive = 0
        with th.no_grad():
            for (batch, labels) in self.test_dataloader:
                batch = batch.float().to(self.config.device)
                labels = labels.float().to(self.config.device)
                outputs = self.model(batch)
                _, predictions = th.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                true_positive += (
                    ((predictions == 1) & (predictions == labels)).sum().item()
                )
                classifier_positive += (predictions == 1).sum().item()
                labelled_positive += (labels == 1).sum().item()
        self.model.train()
        results = Results(
            accuracy=correct / total,
            recall=true_positive / labelled_positive if labelled_positive != 0 else -1,
            precision=true_positive / classifier_positive
            if classifier_positive != 0
            else -1,
        )
        return results


def _make_optimiser(model, config: DictConfig) -> optim.Optimizer:
    optimiser_class = getattr(optim, config.eval.optim_name)
    return optimiser_class(model.parameters(), lr=config.eval.lr)


def _get_in_features(config: DictConfig) -> int:
    # check on what the embedding_generator
    if "_c" in config.model.embedding_generator:
        return config.model.c_size
    elif "_z" in config.model.embedding_generator:
        return config.model.z_size
    else:
        raise Exception("Unable to infer the embeddings size")


def evaluate(
    config: DictConfig,
    embeddings: TensorType["batch", "size"],
    labels: TensorType["batch"],
    global_step: int,
) -> None:
    all_data = EmbeddingsDataset(embeddings, labels)
    train_length = int(config.eval.train_split * len(all_data))
    test_length = len(all_data) - train_length
    assert train_length + test_length == len(all_data)
    train_data, test_data = random_split(all_data, [train_length, test_length])
    train_dataloader = DataLoader(
        train_data, batch_size=config.eval.train_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config.eval.test_batch_size, shuffle=True
    )

    model = SimpleClassifier(
        in_features=_get_in_features(config), n_classes=config.dataset.num_classes
    )
    optimiser = _make_optimiser(model, config)
    runner = EvalRunner(model, train_dataloader, test_dataloader, optimiser, config)
    avg_accuracy = 0.0
    for _ in range(config.eval.repeats):
        for epoch in range(config.eval.epochs):
            LOG.info(f"Starting epoch {epoch}")
            runner.train()
            accuracy = runner.test().accuracy
            LOG.info(f"Accuracy was {accuracy}")
        accuracy = runner.test()
        avg_accuracy += accuracy
        LOG.info(f"Final accuracy was {accuracy}")
    summary_writer = get_summary_writer()
    summary_writer.add_scalar(
        "classifier_accuracy",
        avg_accuracy / config.eval.repeats,
        global_step=global_step,
    )
