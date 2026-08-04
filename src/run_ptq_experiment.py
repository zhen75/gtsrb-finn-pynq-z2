import argparse
import torch
from brevitas.graph.calibrate import calibration_mode
from pathlib import Path
from brevitas.export import export_qonnx
from quant_vgg import QuantVgg
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from gtsrb_utils import get_gtsrb_loaders


def calibrate(ckpt_path, loader, model):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, _) in enumerate(loader):
                images = images.to(device)
                _ = model(images)

                if i >= 15:
                    break


def generate_path(args):
    target = get_next_available_file()
    target.mkdir(parents=True, exist_ok=True)

    template_onnx = "{a}{b}_ptq.onnx"
    template_pth = "{a}{b}_ptq_calibrate.pth"

    onnx_file_name = template_onnx.format(a=args.weight, b=args.activate)
    pth_file_name = template_pth.format(a=args.weight, b=args.activate)

    pth_path = target / pth_file_name
    onnx_path = target / onnx_file_name

    return onnx_path, pth_path, target


def evaluate(pth_path, target, loader, model):

    state_dict = torch.load(
        pth_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    logger = CSVLogger(save_dir=str(target), name="quant_vgg_test")
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    results = trainer.test(
        model=model,
        dataloaders=loader,
    )


def save_result(model, onnx_path, pth_path):
    model.eval()
    current_device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 48, 48,device=current_device)

    export_qonnx(
        model,
        args=dummy_input,
        export_path=str(onnx_path),
        dynamo=True,
        optimize=True,
    )
    torch.save(model.state_dict(), str(pth_path))

def get_args():
    parser = argparse.ArgumentParser(
        description="Brevitas 模型训练后量化 (PTQ) 与验证导出脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--weight", 
        type=str,
        default="w8", 
        choices=["w8", "w4", "w2"], 
        help="指定网络权重的量化位宽 (w8: 8-bit, w4: 4-bit, w2: 2-bit)"
    )
    
    parser.add_argument(
        "--activate", 
        type=str,
        default="a8", 
        choices=["a8", "a4", "a2"], 
        help="指定网络激活值的量化位宽 (a8: 8-bit, a4: 4-bit, a2: 2-bit)"
    )

    return parser.parse_args()


def translate(args):
    weight_map = {
        "w8": 8,
        "w4": 4,
        "w2": 2
    }
    
    activate_map = {
        "a8": 8,
        "a4": 4,
        "a2": 2
    }

    return weight_map[args.weight], activate_map[args.activate]


def get_next_available_file(base_folder="ptq_experiments", folder="experiment"):
    base_path = Path(base_folder)
    counter = 1

    while True:
        target_path = base_path / f"{folder}_{counter}"

        if not target_path.exists():
            return target_path

        counter += 1


def main():
    args = get_args()
    _, _, test_loader, _ = get_gtsrb_loaders(
        batch_size=1024,
        num_workers=0,
    )
    onnx_path, pth_path, target = generate_path(args)
    ckpt_path = "checkpoints/tiny_vgg-best-epoch=37-val_acc=0.981.ckpt"
    train_loader, _, _, _ = get_gtsrb_loaders()
    weight, activate = translate(args)
    model = QuantVgg(weight_bit=weight, activate_bit=activate)
    calibrate(ckpt_path, train_loader, model)
    save_result(model, onnx_path, pth_path)
    evaluate(pth_path, target, test_loader, model)


if __name__ == "__main__":
    main()
