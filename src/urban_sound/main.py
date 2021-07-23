from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import logging
import hydra
from omegaconf import DictConfig
from urban_sound.datasets import get_dataset
from torch import optim, cuda
from urban_sound.model.cpc import CPC
from urban_sound.train import train

LOG = logging.getLogger(__name__)


def _make_optimiser(config: DictConfig) -> Optimizer:
    optimiser_class = getattr(optim, config.optim.name)
    return optimiser_class(lr=config.optim.lr)


def _add_device_to_config(config: DictConfig) -> None:
    config.device = "cuda" if cuda.is_available() else "cpu"


@hydra.main(config_path="config", config_name="debug")
def main(config: DictConfig) -> None:
    dataset = get_dataset(config)
    _add_device_to_config(config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=config.shuffle
    )
    optimiser = _make_optimiser(config)
    model = CPC(config)
    for epoch in range(config.epochs):
        LOG.info(f"Starting epoch {epoch}")
        train(model, dataloader, optimiser, config)


if __name__ == "__main__":
    main()
