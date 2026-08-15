import argparse
import lightning as L
import torch
from brevitas.graph.calibrate import calibration_mode
from pathlib import Path
from brevitas.export import export_qonnx
from quant_vgg import QuantVgg
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from gtsrb_utils import get_gtsrb_loaders
from train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    file_name = f"{args.mode}_experiments"
    target = get_next_available_file(base_folder=PROJECT_ROOT / file_name)
    target.mkdir(parents=True, exist_ok=True)

    template_onnx = "{name}_{mode}.onnx"
    template_pth = "{name}_{mode}.pth"

    experiment_name = args.name or precision_name(*translate(args))
    onnx_file_name = template_onnx.format(name=experiment_name, mode=args.mode)
    pth_file_name = template_pth.format(name=experiment_name, mode=args.mode)

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
    model.to("cpu")
    dummy_input = torch.randn(1, 3, 48, 48, device="cpu")

    export_qonnx(
        model,
        args=dummy_input,
        export_path=str(onnx_path),
        dynamo=False,
    )
    torch.save(model.state_dict(), str(pth_path))


def get_args():
    parser = argparse.ArgumentParser(
        description="Brevitas 模型训练后量化 (PTQ) 与验证导出脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--weight",
        type=str,
        default="8",
        help="7 个权重量化层的位宽，例如 8 或 8,1,1,1,1,1,8",
    )

    parser.add_argument(
        "--activate",
        type=str,
        default="8",
        help="6 个 QuantReLU 层的位宽，例如 8 或 8,1,1,1,1,8",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="ptq",
        choices=["ptq", "qat"],
        help="指定量化方法PTQ或者QAT",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验和导出文件的名称；省略时从位宽列表生成",
    )

    return parser.parse_args()


def parse_bit_widths(value, count, argument_name):
    value = value.lower().replace("w", "").replace("a", "")
    bit_widths = [int(bit) for bit in value.split(",")]

    if len(bit_widths) == 1:
        bit_widths *= count
    if len(bit_widths) != count:
        raise ValueError(f"{argument_name} must contain 1 or {count} values")
    if any(bit not in {1, 2, 4, 8} for bit in bit_widths):
        raise ValueError(f"{argument_name} values must be one of 1, 2, 4, or 8")

    return bit_widths


def precision_name(weight_bits, activate_bits):
    if len(set(weight_bits)) == 1 and len(set(activate_bits)) == 1:
        return f"w{weight_bits[0]}a{activate_bits[0]}"
    return "w" + "-".join(map(str, weight_bits)) + "_a" + "-".join(
        map(str, activate_bits)
    )


def translate(args):
    return (
        parse_bit_widths(args.weight, 7, "--weight"),
        parse_bit_widths(args.activate, 6, "--activate"),
    )


def get_next_available_file(base_folder, folder="experiment"):
    base_path = Path(base_folder)
    counter = 1

    while True:
        target_path = base_path / f"{folder}_{counter}"

        if not target_path.exists():
            return target_path

        counter += 1


def main():
    L.seed_everything(2026, workers=True)
    args = get_args()
    data_root = PROJECT_ROOT / "data"
    _, _, test_loader, _ = get_gtsrb_loaders(
        data_root=data_root,
        batch_size=1024,
        num_workers=0,
    )

    train_loader, val_loader, _, class_weights = get_gtsrb_loaders(
        data_root=data_root, num_workers=2
    )

    onnx_path, pth_path, target = generate_path(args)
    ckpt_path = (
        PROJECT_ROOT / "checkpoints" / "tiny_vgg-best-epoch=37-val_acc=0.981.ckpt"
    )

    weight, activate = translate(args)
    model = QuantVgg(
        weight_bit=weight,
        activate_bit=activate,
        class_weights=class_weights,
    )

    if args.mode == "ptq":
        print("start calibrate")
        calibrate(ckpt_path, train_loader, model)
        print("calibrate complete")
    elif args.mode == "qat":
        print("start QAT")
        calibrate(ckpt_path, train_loader, model)
        best_model_path = train(model, train_loader, val_loader, 60, target)
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("QAT complet")

    save_result(model, onnx_path, pth_path)
    evaluate(pth_path, target, test_loader, model)


if __name__ == "__main__":
    main()
