import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
import gtsrb_utils
from tiny_vgg import TinyVgg


checkpoint_path = "checkpoints/tiny_vgg-best-epoch=37-val_acc=0.981.ckpt"

_, _, test_loader, _ = gtsrb_utils.get_gtsrb_loaders(num_workers=0)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

class_weights = checkpoint["state_dict"]["class_weights"]
logger = CSVLogger(save_dir="logs",name = "tiny_vgg_test")

model = TinyVgg.load_from_checkpoint(
    checkpoint_path,
    class_weights=class_weights,
    map_location="cpu",
)

trainer = Trainer(
    accelerator="auto",
    devices=1,
    logger=logger,
    enable_checkpointing=False,
)

trainer.test(
    model=model,
    dataloaders=test_loader,
)
