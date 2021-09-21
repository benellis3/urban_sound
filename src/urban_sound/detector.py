from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from urban_sound.datasets import get_dataset
from torch import optim
import hydra
from omegaconf import DictConfig
import logging
from urban_sound.datasets.load_data import ElephantImageCollateFunction
from urban_sound.evaluate import EvalRunner
from urban_sound.logging.log import get_summary_writer
from urban_sound.main import (
    _add_device_to_config,
    _add_number_channels_to_config,
    _set_multiprocessing_start_method,
)
from urban_sound.model.detect import get_model

LOG = logging.getLogger(__name__)


def _make_optimiser(model, config: DictConfig):
    optimiser_class = getattr(optim, config.detect.optim_name)
    return optimiser_class(model.parameters(), lr=config.detect.lr)


@hydra.main(config_path="config", config_name="detect")
def detect_rumbles(config: DictConfig) -> None:
    _add_device_to_config(config)
    if config.device == "cuda":
        _set_multiprocessing_start_method()
    all_data = get_dataset(config)
    _add_number_channels_to_config(all_data, config)
    train_length = int(config.detect.train_split * len(all_data))
    test_length = len(all_data) - train_length
    assert train_length + test_length == len(all_data)
    train_data, test_data = random_split(all_data, [train_length, test_length])
    train_dataloader = DataLoader(
        train_data,
        batch_size=config.detect.train_batch_size,
        shuffle=True,
        num_workers=config.detect.workers,
        collate_fn=ElephantImageCollateFunction(config)
        if getattr(config.dataset, "use_collate_fn", False)
        else None,
        prefetch_factor=config.detect.prefetch,
        timeout=config.detect.timeout,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config.detect.test_batch_size,
        shuffle=True,
        num_workers=config.detect.workers,
        collate_fn=ElephantImageCollateFunction(config)
        if getattr(config.dataset, "use_collate_fn", False)
        else None,
        prefetch_factor=config.detect.prefetch,
        timeout=config.detect.timeout,
    )

    model = get_model(config)
    optimiser = _make_optimiser(model, config)
    runner = EvalRunner(model, train_dataloader, test_dataloader, optimiser, config)
    for epoch in range(config.detect.epochs):
        LOG.info(f"Starting epoch {epoch}")
        runner.train()
        results = runner.test()
        summary_writer = get_summary_writer()
        summary_writer.add_scalar(
            "detect/accuracy", results.accuracy, global_step=runner.t
        )
        summary_writer.add_scalar(
            "detect/precision", results.precision, global_step=runner.t
        )
        summary_writer.add_scalar("detect/recall", results.recall, global_step=runner.t)


if __name__ == "__main__":
    detect_rumbles()
