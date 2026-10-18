"""Small CPU training integration test for PCL."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.pcl import PCL


class TinyEncoder(nn.Module):
    """Small image encoder used for PCL integration tests."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 16)

    def forward(self, x):
        return self.linear(self.pool(x).flatten(1))


class TinyDataset(Dataset):
    """Synthetic dataset providing two augmented views and sample indices."""

    def __len__(self):
        return 16

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)

        image = torch.randn(3, 8, 8, generator=generator)

        return {
            "view1": image,
            "view2": image + 0.05,
            "indices": torch.tensor(index, dtype=torch.long),
        }


@pytest.mark.unit
def test_pcl_lightning_training_updates_ema(tmp_path):
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
        temperature=0.2,
    )

    model.optim = {
        "optimizer": {
            "type": "SGD",
            "lr": 0.1,
        },
    }

    student_before = [p.detach().clone() for p in model.backbone.student.parameters()]
    teacher_before = [p.detach().clone() for p in model.backbone.teacher.parameters()]

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[TeacherStudentCallback(verbose=False)],
        default_root_dir=tmp_path,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            TinyDataset(),
            batch_size=4,
            shuffle=False,
            num_workers=0,
        ),
    )

    assert trainer.global_step == 4

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            student_before,
            model.backbone.student.parameters(),
        )
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    assert model.queue.get().shape[0] == 16
