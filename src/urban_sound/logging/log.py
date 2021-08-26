from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from functools import lru_cache
from torchtyping import TensorType
from sklearn.manifold import TSNE
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import getcwd
from pathlib import Path


@lru_cache(maxsize=1)
def get_summary_writer():
    return SummaryWriter()


def log_tsne(
    embeddings: TensorType["N", "dims"],
    config: DictConfig,
    t: int,
    labels=None,
    display=False,
    label_map=None,
) -> None:
    tsne = TSNE(
        perplexity=config.tsne.perplexity,
        n_iter=config.tsne.n_iter,
        learning_rate=config.tsne.learning_rate,
    )
    clusters = tsne.fit_transform(embeddings)
    # plot the clusters
    fig, _ = plt.subplots(1, 1)
    if labels is None:
        data = pd.DataFrame(data=clusters, columns=["x", "y"])
        sns.scatterplot(x="x", y="y", data=data)
    else:
        data = pd.DataFrame(
            data=np.concatenate([clusters, labels], axis=1), columns=["x", "y", "label"]
        )
        if label_map:
            data = data.replace({"label": label_map})
        sns.scatterplot(x="x", y="y", hue="label", data=data)

    if display:
        plt.savefig(Path(getcwd()) / f"tsne_{t}.pdf")
        plt.show(block=True)
    summary_writer = get_summary_writer()
    summary_writer.add_figure("tsne", fig, global_step=t)
