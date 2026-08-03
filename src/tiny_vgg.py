import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score


class TinyVgg(L.LightningModule):
    def __init__(self, lr=1e-3, class_weights=None, dropout=0.2):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

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

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 43),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)

        return x

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
