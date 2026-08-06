import torch
import torch.nn as nn
import brevitas.onnx as bo
from gtsrb_utils import get_gtsrb_loaders
from quant_vgg import QuantVgg

train_loader,_,_,_ = get_gtsrb_loaders()
pth_path = "quant_onnx/quantized_vgg_calibrated.pth"
export_path = "quant_onnx/quantized_vgg_for_finn.onnx"
ckpt_path = "checkpoints/tiny_vgg-best-epoch=37-val_acc=0.981.ckpt"

checkpoint = torch.load(ckpt_path,map_location=torch.device("cpu"))

state_dict = checkpoint["state_dict"]

model =QuantVgg()

model.load_state_dict(state_dict, strict=False)

model.train()

for module in model.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.Dropout)):
        module.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for i,(images,_) in enumerate(train_loader):
        images = images.to(device)
        _ = model(images)

        if i >= 15:
            break

model.eval()
model.cpu()

dummy_input = torch.randn(1, 3, 48, 48, device="cpu")

torch.save(model.state_dict(), pth_path)
bo.export_qonnx(model, dummy_input, export_path)
