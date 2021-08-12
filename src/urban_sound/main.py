from typing import Dict
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import logging
import hydra
from omegaconf import DictConfig
from torch.utils.data.dataset import Dataset
from urban_sound.datasets import get_dataset
from torch import optim, cuda
from urban_sound.logging.log import get_summary_writer
from urban_sound.model.cpc import CPC
from urban_sound.train import Runner

LOG = logging.getLogger(__name__)


def _make_optimiser(model, config: DictConfig) -> Optimizer:
    optimiser_class = getattr(optim, config.optim.name)
    return optimiser_class(model.parameters(), lr=config.optim.lr)


def _add_device_to_config(config: DictConfig) -> None:
    config.device = "cuda" if cuda.is_available() else "cpu"


def _add_number_channels_to_config(dataset: Dataset, config: DictConfig) -> None:
    """gets the number of channels the dataset has and adds them to the
    config. This assumes that the data has the shape (C, L) where C is the
    number of channels and L is the length.

    Args:
        dataset (Dataset): the dataset to use
    """
    config.dataset.channels = dataset[0][0].size(0)


def close_summary_writer() -> None:
    summary_writer = get_summary_writer()
    summary_writer.flush()
    summary_writer.close()


@hydra.main(config_path="config", config_name="debug")
def main(config: DictConfig) -> None:
    dataset = get_dataset(config)
    _add_device_to_config(config)
    _add_number_channels_to_config(dataset, config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=config.shuffle
    )
    model = CPC(config)
    optimiser = _make_optimiser(model, config)
    runner = Runner(model, dataloader, optimiser, config)
    for epoch in range(config.epochs):
        LOG.info(f"Starting epoch {epoch}")
        runner.train()
    close_summary_writer()


if __name__ == "__main__":
    main()
