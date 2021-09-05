import torch as th
from unittest.mock import Mock, patch

from urban_sound.train import compute_cpc_loss, _compute_accuracy, Runner
from math import log, exp, isclose

RTOL = 1e-05
ATOL = 1e-08


def test_score_computation():

    log_scores = th.tensor(
        [[[1, 4], [2, 5], [5, 8]], [[0, 3], [17, 20], [14, 17]]], dtype=th.float32
    )
    # all the pairs have a difference of 3, so can divide top
    # and bottom of the fraction by the maximum to get this answer
    loss = -log(1.0 / (1.0 + exp(3)))

    scores = Mock(
        pos=log_scores[:, :, 0].unsqueeze(-1), neg=log_scores[:, :, 1].unsqueeze(-1)
    )
    with th.no_grad():
        computed_loss = compute_cpc_loss(scores)
        assert isclose(loss, computed_loss, rel_tol=RTOL, abs_tol=ATOL)


def test_train():
    # test that train makes the correct calls
    model = Mock()
    optimiser = Mock()
    loss = Mock()
    config = Mock(
        log_output=False, training=Mock(update_interval=100, tsne_interval=1000)
    )
    # want to verify call order among the model, the optimiser and the loss
    with patch("urban_sound.train.compute_cpc_loss", return_value=loss):
        runner = Runner(model, [(Mock(), Mock())], optimiser, config)
        runner.train()
        model.assert_called_once()
        optimiser.zero_grad.assert_called_once()
        loss.backward.assert_called_once()
        optimiser.step.assert_called_once()


def test_accuracy():

    log_scores = th.tensor(
        [[[4, 1], [8, 5], [5, 8]], [[9, 3], [17, 20], [14, 17]]], dtype=th.float32
    )
    accuracy = 3.0 / 6.0
    scores = Mock(
        pos=log_scores[:, :, 0].unsqueeze(-1), neg=log_scores[:, :, 1].unsqueeze(-1)
    )
    computed_accuracy = _compute_accuracy(scores)
    assert isclose(accuracy, computed_accuracy, rel_tol=RTOL, abs_tol=ATOL)
