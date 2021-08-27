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
from umap import UMAP


@lru_cache(maxsize=1)
def get_summary_writer():
    return SummaryWriter()


def get_dim_reducer(config):
    if config.dim_reduction.name == "tsne":
        return TSNE(
            perplexity=config.dim_reduction.perplexity,
            n_iter=config.dim_reduction.n_iter,
            learning_rate=config.dim_reduction.learning_rate,
        )
    elif config.dim_reduction.name == "umap":
        return UMAP(
            n_neighbors=config.dim_reduction.n_neighbours,
            n_components=config.dim_reduction.n_components,
            min_dist=config.dim_reduction.min_dist,
        )
    else:
        raise Exception("Do not recognise dimensionality reduction param")


def log_tsne(
    embeddings: TensorType["N", "dims"],
    config: DictConfig,
    t: int,
    tag=None,
    labels=None,
    display=False,
    label_map=None,
) -> None:
    dim_reducer = get_dim_reducer(config)
    clusters = dim_reducer.fit_transform(embeddings)
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
        fname = f"tsne_{t}.pdf" if not tag else f"tsne_{tag}_{t}.pdf"
        plt.savefig(Path(getcwd()) / fname)
        plt.show(block=True)
    summary_writer = get_summary_writer()
    summary_writer.add_figure("tsne", fig, global_step=t)
