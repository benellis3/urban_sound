from typing import Dict
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import load, optim, cuda, device
import logging
from multiprocessing import set_start_method
import hydra
from omegaconf import DictConfig
from torch.utils.data.dataset import Dataset
from urban_sound.datasets import get_dataset
from urban_sound.logging.log import get_summary_writer
from urban_sound.model.cpc import CPC
from urban_sound.train import Runner

LOG = logging.getLogger(__name__)


def _make_optimiser(model, config: DictConfig) -> Optimizer:
    optimiser_class = getattr(optim, config.optim.name)
    return optimiser_class(model.parameters(), lr=config.optim.lr)


def _add_device_to_config(config: DictConfig) -> None:
    config.device = "cuda" if cuda.is_available() else "cpu"


def _set_multiprocessing_start_method() -> None:
    set_start_method("spawn")


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


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    _add_device_to_config(config)
    if config.device == "cuda":
        _set_multiprocessing_start_method()
    dataset = get_dataset(config)
    _add_number_channels_to_config(dataset, config)
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle
    )
    model = CPC(config)
    if config.saved_model:
        model.load_state_dict(
            load(config.saved_model, map_location=device(config.device))
        )

    optimiser = _make_optimiser(model, config)
    runner = Runner(model, dataloader, optimiser, config)
    if not config.generate_tsne_only:
        for epoch in range(config.training.epochs):
            LOG.info(f"Starting epoch {epoch}")
            runner.train()
        close_summary_writer()
    else:
        runner.generate_tsne_embeddings(display=True)
        close_summary_writer()


if __name__ == "__main__":
    main()
