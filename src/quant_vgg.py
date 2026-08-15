import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score
import brevitas.nn as qnn
import brevitas.quant as bq
from brevitas.quant import Int32Bias
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import SignedBinaryActPerTensorConst
from brevitas.quant import SignedBinaryWeightPerTensorConst


WEIGHT_LAYER_COUNT = 7
ACTIVATION_LAYER_COUNT = 6


def _as_bit_widths(bit_widths, expected_count, name):
    if isinstance(bit_widths, int):
        bit_widths = [bit_widths] * expected_count
    else:
        bit_widths = list(bit_widths)

    if len(bit_widths) != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} values, got {len(bit_widths)}"
        )

    return bit_widths


def _weight_quantizer(bit_width):
    if bit_width == 1:
        return SignedBinaryWeightPerTensorConst
    return Int8WeightPerChannelFloat.let(bit_width=bit_width)


def _activation_quantizer(bit_width):
    if bit_width == 1:
        return SignedBinaryActPerTensorConst
    return Uint8ActPerTensorFloat.let(bit_width=bit_width)


def _quant_activation_layer(bit_width, return_quant_tensor=False):
    activation_class = qnn.QuantHardTanh if bit_width == 1 else qnn.QuantReLU
    return activation_class(
        act_quant=_activation_quantizer(bit_width),
        return_quant_tensor=return_quant_tensor,
    )


class QuantVgg(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        class_weights=None,
        dropout=0.2,
        weight_bit=(8,) * WEIGHT_LAYER_COUNT,
        activate_bit=(8,) * ACTIVATION_LAYER_COUNT,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        weight_bits = _as_bit_widths(
            weight_bit, WEIGHT_LAYER_COUNT, "weight_bit"
        )
        activate_bits = _as_bit_widths(
            activate_bit, ACTIVATION_LAYER_COUNT, "activate_bit"
        )
        quant_weights = [_weight_quantizer(bit) for bit in weight_bits]

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.dropout = dropout

        self.train_acc = Accuracy(task="multiclass", num_classes=43)

        self.val_acc = Accuracy(task="multiclass", num_classes=43)

        self.val_f1 = F1Score(task="multiclass", num_classes=43, average="macro")

        self.test_acc = Accuracy(task="multiclass", num_classes=43)

        self.test_f1 = F1Score(task="multiclass", num_classes=43, average="macro")

        self.quant_input = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.block1 = nn.Sequential(
            qnn.QuantConv2d(
                3, 32, 3, padding=1, bias=False, weight_quant=quant_weights[0]
            ),
            nn.BatchNorm2d(32),
            _quant_activation_layer(activate_bits[0]),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            qnn.QuantConv2d(
                32, 32, 3, padding=1, bias=False, weight_quant=quant_weights[1]
            ),
            nn.BatchNorm2d(32),
            _quant_activation_layer(activate_bits[1]),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            qnn.QuantConv2d(
                32, 64, 3, padding=1, bias=False, weight_quant=quant_weights[2]
            ),
            nn.BatchNorm2d(64),
            _quant_activation_layer(activate_bits[2]),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            qnn.QuantConv2d(
                64, 64, 3, padding=1, bias=False, weight_quant=quant_weights[3]
            ),
            nn.BatchNorm2d(64),
            _quant_activation_layer(activate_bits[3]),
            nn.MaxPool2d(2),
        )

        self.block5 = nn.Sequential(
            qnn.QuantConv2d(
                64,
                128,
                3,
                padding=1,
                bias=False,
                weight_quant=quant_weights[4],
            ),
            nn.BatchNorm2d(128),
            _quant_activation_layer(activate_bits[4], return_quant_tensor=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                1152, 256, weight_quant=quant_weights[5], bias_quant=Int32Bias
            ),
            _quant_activation_layer(activate_bits[5], return_quant_tensor=True),
            nn.Dropout(self.dropout),
            qnn.QuantLinear(
                256, 43, weight_quant=quant_weights[6], bias_quant=Int32Bias
            ),
        )

    def forward(self, x):
        x = self.quant_input(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)

        return x

    def on_train_epoch_start(self):
        if self.current_epoch >= 2:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.train_acc(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        loss = F.cross_entropy(logits, y)

        self.val_acc(logits, y)

        self.val_f1(logits, y)

        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_acc, "val_f1": self.val_f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        loss = F.cross_entropy(logits, y)

        self.test_acc(logits, y)

        self.test_f1(logits, y)

        self.log_dict(
            {"test_loss": loss, "test_acc": self.test_acc, "test_f1": self.test_f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
