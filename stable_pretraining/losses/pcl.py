"""Prototype contrastive loss for PCL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNCELoss(nn.Module):
    """ProtoNCE with explicitly selected negative prototypes."""

    def forward(
        self,
        queries: torch.Tensor,
        positive_prototypes: torch.Tensor,
        negative_prototypes: torch.Tensor,
        positive_concentrations: torch.Tensor,
        negative_concentrations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the prototype contrastive objective.

        Shapes:
            queries: [B, D]
            positive_prototypes: [B, D]
            negative_prototypes: [K, D]
            positive_concentrations: [B]
            negative_concentrations: [K]
        """
        if not (
            torch.all(positive_concentrations > 0)
            and torch.all(negative_concentrations > 0)
        ):
            raise ValueError("Concentrations must be positive.")

        positive_logits = (queries * positive_prototypes).sum(
            dim=-1, keepdim=True
        ) / positive_concentrations[:, None]

        negative_logits = (queries @ negative_prototypes.T) / negative_concentrations[
            None, :
        ]

        logits = torch.cat([positive_logits, negative_logits], dim=1)

        targets = torch.zeros(
            queries.size(0),
            dtype=torch.long,
            device=queries.device,
        )

        return F.cross_entropy(logits, targets)


@torch.no_grad()
def select_prototypes(
    prototypes: torch.Tensor,
    assignments: torch.Tensor,
    concentrations: torch.Tensor,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select PCL positive prototypes and batch-shared negatives.

    Args:
        prototypes: Cluster centroids with shape [K, D].
        assignments: Cluster IDs for batch samples, shape [B].
        concentrations: Positive cluster temperatures, shape [K].
        num_negatives: Number of distinct negative prototypes.
        generator: Optional random generator for reproducibility.

    Returns:
        Positive prototypes [B, D],
        negative prototypes [R, D],
        positive concentrations [B],
        negative concentrations [R].
    """
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [K, D]")
    if assignments.ndim != 1 or assignments.dtype != torch.long:
        raise ValueError("assignments must be a 1D long tensor")
    if concentrations.shape != (prototypes.shape[0],):
        raise ValueError("concentrations must have shape [K]")
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")

    num_clusters = prototypes.shape[0]

    if not torch.all((assignments >= 0) & (assignments < num_clusters)).item():
        raise ValueError("Cluster assignment out of range")

    if not torch.isfinite(concentrations).all().item():
        raise ValueError("Concentrations must be finite")
    if not (concentrations > 0).all().item():
        raise ValueError("Concentrations must be positive")

    # Every prototype assigned to a sample in this batch
    # must be excluded from the shared negative set.
    positive_ids = torch.unique(assignments)

    eligible = torch.ones(num_clusters, dtype=torch.bool, device=prototypes.device)
    eligible[positive_ids.to(prototypes.device)] = False

    negative_candidates = torch.arange(num_clusters, device=prototypes.device)[eligible]

    if num_negatives > negative_candidates.numel():
        raise ValueError("Not enough negative clusters outside batch positives")

    # Sample without replacement.
    order = torch.randperm(
        negative_candidates.numel(),
        device=prototypes.device,
        generator=generator,
    )
    negative_ids = negative_candidates[order[:num_negatives]]

    positive_ids = assignments.to(prototypes.device)

    return (
        prototypes[positive_ids],
        prototypes[negative_ids],
        concentrations[positive_ids],
        concentrations[negative_ids],
    )
