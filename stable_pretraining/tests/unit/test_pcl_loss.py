"""Tests for the PCL prototype loss."""

import pytest
import torch

from stable_pretraining.losses.pcl import (
    ProtoNCELoss,
    select_prototypes,
)


@pytest.mark.unit
def test_protonce_loss_backward():
    queries = torch.randn(4, 8, requires_grad=True)

    positive_prototypes = torch.randn(4, 8)
    negative_prototypes = torch.randn(6, 8)

    positive_concentrations = torch.ones(4)
    negative_concentrations = torch.ones(6)

    loss_fn = ProtoNCELoss()

    loss = loss_fn(
        queries,
        positive_prototypes,
        negative_prototypes,
        positive_concentrations,
        negative_concentrations,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()


@pytest.mark.unit
def test_protonce_matches_manual_calculation():
    """Compare ProtoNCE with an independent manual calculation."""
    queries = torch.tensor([[1.0, 0.0]], requires_grad=True)

    positive = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    pos_concentration = torch.tensor([0.5])
    neg_concentrations = torch.tensor([0.5, 0.5])

    loss = ProtoNCELoss()(
        queries,
        positive,
        negatives,
        pos_concentration,
        neg_concentrations,
    )

    # Positive similarity = 1; negative similarities = 0 and -1.
    # Temperature/concentration = 0.5.
    # Resulting logits = [2, 0, -2].
    expected = -torch.log(
        torch.exp(torch.tensor(2.0))
        / (
            torch.exp(torch.tensor(2.0))
            + torch.exp(torch.tensor(0.0))
            + torch.exp(torch.tensor(-2.0))
        )
    )

    torch.testing.assert_close(loss, expected)


@pytest.mark.unit
def test_select_prototypes_excludes_batch_positives():
    # Five identifiable prototype vectors.
    prototypes = torch.eye(5)
    concentrations = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Three samples belong to clusters 2, 0, and 3.
    assignments = torch.tensor([2, 0, 3])

    pos, neg, pos_temp, neg_temp = select_prototypes(
        prototypes,
        assignments,
        concentrations,
        num_negatives=2,
        generator=torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(pos, prototypes[assignments])
    torch.testing.assert_close(pos_temp, concentrations[assignments])

    # Only clusters 1 and 4 are eligible negatives.
    selected_ids = neg.argmax(dim=1).tolist()
    assert set(selected_ids) == {1, 4}
    assert set(selected_ids).isdisjoint({0, 2, 3})

    torch.testing.assert_close(neg_temp, concentrations[selected_ids])


@pytest.mark.unit
def test_select_prototypes_insufficient_negatives():
    prototypes = torch.eye(3)
    assignments = torch.tensor([0, 1, 2])
    concentrations = torch.ones(3)

    with pytest.raises(ValueError, match="Not enough negative"):
        select_prototypes(
            prototypes,
            assignments,
            concentrations,
            num_negatives=1,
        )
