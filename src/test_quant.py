import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
import gtsrb_utils
from quant_vgg import QuantVgg

weights_path = "quant_onnx/quantized_vgg_calibrated.pth"

_, _, test_loader, _ = gtsrb_utils.get_gtsrb_loaders(
    batch_size=1024,
    num_workers=0,
)

model = QuantVgg()
state_dict = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict, strict=True)
logger = CSVLogger(save_dir="logs",name = "quant_vgg_test")
trainer = Trainer(
    accelerator="auto",
    devices=1,
    logger=logger,
    enable_checkpointing=False,
)

results = trainer.test(
    model=model,
    dataloaders=test_loader,
)
