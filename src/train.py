from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from tiny_vgg import TinyVgg
import gtsrb_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

L.seed_everything(2026)

train_loader, val_loader, test_loader, class_weights = gtsrb_utils.get_gtsrb_loaders()

model = TinyVgg(class_weights=class_weights,dropout=0.5)

logger = TensorBoardLogger(save_dir="logs", name="tiny_vgg")

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    dirpath="checkpoints",
    filename="tiny_vgg-best-{epoch:02d}-{val_acc:.3f}",
    save_top_k=1,
)

f1_checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    dirpath="checkpoints",
    filename="tiny_vgg-best-f1-{epoch:02d}-{val_f1:.3f}",
    save_top_k=1,
)

trainer = Trainer(
    max_epochs=50,
    logger=logger,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback, f1_checkpoint_callback],
)


trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

trainer.test(
    model=model,
    dataloaders=test_loader,
    ckpt_path=checkpoint_callback.best_model_path,
)
