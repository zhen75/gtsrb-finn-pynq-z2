import torch
from pathlib import Path
from torch import nn
from brevitas.export import export_qonnx

class GTSRBPreprocessNHWC(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.3399, 0.3121, 0.3214],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.2760, 0.2625, 0.2690],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x = x / 255.0
        x = (x - self.mean) / self.std

        return x


def save_result(model, onnx_path):
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 48, 48, 3, device="cpu")

    target = onnx_path / "preprocessor.onnx"

    export_qonnx(
        model,
        args=dummy_input,
        export_path=str(target),
        dynamo=False,
    )


if __name__ == "__main__":
    model = GTSRBPreprocessNHWC()
    file_path = Path("output_preprocessor_onnx")
    file_path.mkdir(parents=True, exist_ok=True)
    save_result(model, file_path)
