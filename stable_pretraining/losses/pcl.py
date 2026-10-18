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

        positive_logits = (
            (queries * positive_prototypes).sum(dim=-1, keepdim=True)
            / positive_concentrations[:, None]
        )

        negative_logits = (
            queries @ negative_prototypes.T
        ) / negative_concentrations[None, :]

        logits = torch.cat([positive_logits, negative_logits], dim=1)

        targets = torch.zeros(
            queries.size(0),
            dtype=torch.long,
            device=queries.device,
        )

        return F.cross_entropy(logits, targets)