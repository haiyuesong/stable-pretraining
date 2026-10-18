"""Tests for the PCL warm-up model."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.methods.pcl import PCL


class TinyEncoder(nn.Module):
    """Small encoder for fast CPU tests."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 16)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.linear(x)


def make_model():
    return PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
        temperature=0.2,
    )


@pytest.mark.unit
def test_pcl_warmup_forward_and_backward():
    torch.manual_seed(7)
    model = make_model()
    model.train()

    # Populate the negative queue before testing gradients.
    initial_keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(initial_keys)

    view1 = torch.randn(4, 3, 8, 8)
    view2 = torch.randn(4, 3, 8, 8)

    output = model(view1, view2)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.queries.shape == (4, 8)
    assert output.keys.shape == (4, 8)
    assert model.queue.get().shape[0] == 12

    output.loss.backward()

    student_grads = [p.grad for p in model.backbone.student.parameters()]

    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in student_grads
    )

    assert all(p.grad is None for p in model.backbone.teacher.parameters())

    assert all(p.grad is None for p in model.projector.teacher.parameters())


@pytest.mark.unit
def test_pcl_inference_does_not_modify_queue():
    model = make_model()
    model.eval()

    images = torch.randn(2, 3, 8, 8)
    old_queue = model.queue.get().clone()

    output = model(images)

    assert output.embedding.shape == (2, 16)
    assert output.embedding.requires_grad is False
    torch.testing.assert_close(model.queue.get(), old_queue)


@pytest.mark.unit
def test_pcl_empty_queue_warmup():
    model = make_model()
    model.train()

    output = model(
        torch.randn(4, 3, 8, 8),
        torch.randn(4, 3, 8, 8),
    )

    assert torch.isfinite(output.loss)
    torch.testing.assert_close(
        output.loss,
        torch.tensor(0.0),
    )
    assert model.queue.get().shape[0] == 4


@pytest.mark.unit
def test_pcl_combines_instance_and_prototype_losses():
    torch.manual_seed(7)

    model = make_model()
    model.train()

    # Populate the negative queue.
    initial_keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(initial_keys)

    # Five dataset samples, assigned to five clusters.
    assignments = torch.tensor([0, 1, 2, 3, 4])

    prototypes = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)

    concentrations = torch.full((5,), 0.2)

    model.set_clustering(
        assignments,
        prototypes,
        concentrations,
    )

    # Batch samples 0 and 2.
    indices = torch.tensor([0, 2])

    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        indices=indices,
    )

    assert output.prototype_loss is not None
    assert torch.isfinite(output.prototype_loss)
    assert torch.isfinite(output.instance_loss)

    torch.testing.assert_close(
        output.loss,
        output.instance_loss + output.prototype_loss,
    )

    output.loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.student.parameters()
    )


@pytest.mark.unit
def test_pcl_optimizer_and_ema_update():
    """Verify student optimization and teacher EMA updates."""
    torch.manual_seed(7)

    model = make_model()
    model.train()

    # Use a populated queue to obtain nonzero contrastive gradients.
    keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(keys)

    optimizer = torch.optim.SGD(
        list(model.backbone.student.parameters())
        + list(model.projector.student.parameters()),
        lr=0.1,
    )

    # Save parameters before training.
    student_before = [p.detach().clone() for p in model.backbone.student.parameters()]
    teacher_before = [p.detach().clone() for p in model.backbone.teacher.parameters()]

    view1 = torch.randn(4, 3, 8, 8)
    view2 = torch.randn(4, 3, 8, 8)

    optimizer.zero_grad()
    output = model(view1, view2)
    output.loss.backward()
    optimizer.step()

    # Student should change after gradient descent.
    student_after = list(model.backbone.student.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(student_before, student_after)
    )

    # Teacher should not be modified by optimizer.step().
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    # Simulate the EMA update normally triggered by
    # TeacherStudentCallback after an optimizer step.
    model.backbone.update_teacher()
    model.projector.update_teacher()

    # Teacher should now change through EMA.
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    assert all(p.grad is None for p in model.backbone.teacher.parameters())


@pytest.mark.unit
def test_pcl_dictionary_training_interface():
    torch.manual_seed(7)
    model = make_model()
    model.train()

    model.queue.append(torch.nn.functional.normalize(torch.randn(8, 8), dim=-1))

    batch = {
        "view1": torch.randn(4, 3, 8, 8),
        "view2": torch.randn(4, 3, 8, 8),
        "indices": torch.arange(4),
    }

    state = model(batch, stage="fit")

    assert isinstance(state, dict)
    assert torch.isfinite(state["loss"])
    assert state["loss"].ndim == 0

    state["loss"].backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.student.parameters()
    )
