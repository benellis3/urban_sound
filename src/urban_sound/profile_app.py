import hydra
from torch.utils.data import DataLoader
from urban_sound.datasets import get_dataset


@hydra.main(config_path="config", config_name="detect")
def main(config):
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=16)
    iterations = 100
    count = 0
    for batch, label in dataloader:
        if count > iterations:
            break
        count += 1

    print(count)


if __name__ == "__main__":
    main()
