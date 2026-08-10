from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from tiny_vgg import TinyVgg
import gtsrb_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path


def train(model, train_loader, val_loader, epochs, path):
    logger = TensorBoardLogger(save_dir="logs", name="tiny_vgg")

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=10, verbose=True, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath=str(path),
        filename="tiny_vgg-best-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1,
    )

    f1_checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        dirpath=str(path),
        filename="tiny_vgg-best-f1-{epoch:02d}-{val_f1:.3f}",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, f1_checkpoint_callback, early_stop_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    L.seed_everything(2026)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    train_loader, val_loader, test_loader, class_weights = (
        gtsrb_utils.get_gtsrb_loaders()
    )

    model = TinyVgg(class_weights=class_weights, dropout=0.2)

    checkpoints_path = PROJECT_ROOT / "checkpoints"

    train(model, train_loader, val_loader, 40, checkpoints_path)
