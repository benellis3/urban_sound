import hydra
from torch.utils.data.dataloader import DataLoader
from urban_sound.datasets import get_dataset
from urban_sound.datasets.load_data import ElephantImageCollateFunction
from urban_sound.main import (
    _add_device_to_config,
    _add_number_channels_to_config,
    _set_multiprocessing_start_method,
)
from pathlib import Path
from pandas import DataFrame
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "elephant_images"


@hydra.main(config_path="config", config_name="detect")
def main(config):
    _add_device_to_config(config)
    if config.device == "cuda":
        _set_multiprocessing_start_method()
    dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.detect.train_batch_size,
        num_workers=config.detect.workers,
        collate_fn=ElephantImageCollateFunction(config)
        if getattr(config.dataset, "use_collate_fn", False)
        else None,
    )
    _add_number_channels_to_config(dataset, config)

    metadata = DataFrame(columns=["index", "label"])

    for (batch, label) in tqdm(dataloader):
        for i, img in enumerate(batch):
            station = dataset._get_metadata_item(i, "station")
            station_dir = DATA_PATH / station
            station_dir.mkdir(exist_ok=True)
            metadata.loc[len(metadata.index)] = [i, label]
            # images are padded to be RGB format so we just
            # extract one channel
            fig = plt.figure()
            plt.imshow(img[0].cpu().numpy(), origin="lower")
            ax = plt.gca()
            ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (
                ax.get_ylim()[1] - ax.get_ylim()[0]
            )
            ax.set_aspect(ratio_default)
            fig.savefig(station_dir / f"{i}.png")
            plt.close()


if __name__ == "__main__":
    main()
