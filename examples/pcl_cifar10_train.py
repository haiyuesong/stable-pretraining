"""Small CIFAR-10 smoke experiment for PCL."""

import os

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.pcl import PCL, PCLClusteringCallback

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback


class IndexedCIFAR10(Dataset):
    """Provide augmented views and stable indices for PCL."""

    def __init__(self, root="./data", subset_size=128):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=False,
        )

        if not 0 < subset_size <= len(self.dataset):
            raise ValueError("Invalid subset size")

        generator = torch.Generator().manual_seed(42)
        self.indices = torch.randperm(len(self.dataset), generator=generator)[
            :subset_size
        ].tolist()

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        self.augment = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.cluster_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]

        return {
            "image": self.cluster_transform(image),
            "view1": self.augment(image),
            "view2": self.augment(image),
            "indices": torch.tensor(index, dtype=torch.long),
        }


class PCLLossLogger(Callback):
    """Record total, instance, and prototype losses."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return

        for name in ("loss", "instance_loss", "prototype_loss"):
            value = outputs.get(name)

            if value is not None:
                pl_module.log(
                    f"pcl/{name}",
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name == "loss"),
                    logger=True,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        parts = [f"epoch={trainer.current_epoch}"]

        for name in ("loss", "instance_loss", "prototype_loss"):
            key = f"pcl/{name}"
            if key in metrics:
                parts.append(f"{name}={metrics[key].item():.4f}")

        print("[PCL Loss] " + ", ".join(parts), flush=True)


def main():
    pl.seed_everything(42)

    dataset = IndexedCIFAR10(subset_size=512)

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    cluster_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = PCL(
        encoder_name="resnet18",
        low_resolution=True,
        projector_dims=(512, 128),
        queue_length=256,
        temperature=0.2,
    )

    model.num_proto_negatives = 2
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.01}}

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator=os.environ.get("PCL_ACCELERATOR", "cpu"),
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        callbacks=[
            TeacherStudentCallback(verbose=False),
            PCLClusteringCallback(
                dataloader=cluster_loader,
                num_clusters=(32, 64),
                warmup_epochs=1,
                seed=42,
            ),
            ModelCheckpoint(
                dirpath="checkpoints/pcl-cifar10-loss-test",
                filename="epoch-{epoch:03d}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=1,
            ),
            PCLLossLogger(),
        ],
    )

    trainer.fit(model, train_dataloaders=train_loader)

    print("PCL CIFAR-10 smoke test completed.")
    print("Optimizer steps:", trainer.global_step)
    print("Clustering levels:", 1 + len(model.additional_clusterings))


if __name__ == "__main__":
    main()
