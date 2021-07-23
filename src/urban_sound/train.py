from omegaconf.dictconfig import DictConfig
from urban_sound.model.cpc import CPC, Scores
from torch.utils.data import DataLoader
from torch import optim
import torch as th
import torch.nn.functional as F
from tqdm import tqdm


def make_optimiser(model, config):
    optimiser_class = getattr(optim, config.optimiser)
    return optimiser_class(model.parameters(), lr=config.lr)


def _compute_accuracy(scores):
    accuracy = scores.pos > scores.neg
    accuracy = th.all(accuracy, -1)
    return th.mean(accuracy.float())


def compute_cpc_loss(log_scores: Scores):
    scores = th.cat([log_scores.pos, log_scores.neg], dim=-1)
    return -th.mean(F.log_softmax(scores, dim=-1)[:, :, 0])


def train(
    model: CPC, dataloader: DataLoader, optimiser: optim.Optimizer, config: DictConfig
) -> None:
    # iterate through the dataloader
    iter = tqdm(enumerate(dataloader), dynamic_ncols=True)
    for i, (batch, _) in iter:
        # compute the scores
        log_scores: Scores = model(batch)
        loss = compute_cpc_loss(log_scores)
        # compute the loss function
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if config.log_output and i % config.update_interval == 0:
            iter.set_postfix({"loss": loss, "accuracy": _compute_accuracy(log_scores)})
