import torch
from torch import nn
import torchmetrics
from typing import Optional, Any
import pytorch_lightning as pl
from dnn_model import DnnHeader


class AudioCnn1d(pl.LightningModule):
    def __init__(
        self,
        fc_layer: bool = True,
        n_class: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        # TRAINING PARAMETERS
        if n_class:
            self.n_class = n_class
        else:
            self.n_class = 2
        self.lr = lr
        self.wd = weight_decay
        self.decision_layer = nn.Softmax(dim=1)
        self.metric = torchmetrics.F1Score()

        dropout = 0.2
        self.cnn_block_0 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                padding=2,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        )

        self.cnn_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=10,
                out_channels=10,
                kernel_size=5,
                padding=2,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        )

        self.cnn_block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=10,
                out_channels=5,
                kernel_size=5,
                padding=2,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        )

        self.model = nn.Sequential(
            self.cnn_block_0,
            self.cnn_block_1,
            self.cnn_block_2,
        )
        if fc_layer:
            self.fc = nn.Linear(200, self.n_class)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, target):
        return nn.CrossEntropyLoss()(y_hat.float(), target.float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return optimizer

    def share_step(self, batch):
        labels = batch["label"]
        embeddings = self(batch).flatten()
        y_hat = self.fc(embeddings)
        y_hat = self.decision_layer(y_hat)
        # reshape_labels = labels.view(labels.shape[0], 1)
        enc_labels = nn.functional.one_hot(labels % self.n_class)
        loss = self.loss(y_hat, enc_labels)
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        # aug_batch = self.augmentation(batch)
        aug_batch = batch["audio"].unsqueeze(dim=1)
        _, loss = self.share_step(aug_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self.share_step(batch)
        logits = torch.argmax(y_hat, dim=1)
        score = self.metric(logits, batch["label"])
        self.log("valid/loss", loss)
        self.log("valid/score", score)
        return loss  # , accu


class AcceleroCnn1d(pl.LightningModule):
    def __init__(
        self,
        fc_layer: bool = True,
        n_class: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        # TRAINING PARAMETERS
        if n_class:
            self.n_class = n_class
        else:
            self.n_class = 2
        self.lr = lr
        self.wd = weight_decay
        self.metric = torchmetrics.F1Score()
        self.decision_layer = nn.Softmax(dim=1)

        dropout = 0.2
        self.cnn_block_0 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=10,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        )

        self.cnn_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=10,
                out_channels=10,
                kernel_size=3,
                padding=5,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
        )

        self.model = nn.Sequential(
            self.cnn_block_0,
            self.cnn_block_1,
        )
        if fc_layer:
            self.fc = nn.Linear(5120, self.n_class)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, target):
        return nn.CrossEntropyLoss()(y_hat.float(), target.float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return optimizer

    def share_step(self, batch):
        labels = batch["label"]
        embeddings = self(batch).flatten()
        y_hat = self.fc(embeddings)
        y_hat = self.decision_layer(y_hat)
        # reshape_labels = labels.view(labels.shape[0], 1)
        enc_labels = nn.functional.one_hot(labels % self.n_class)
        loss = self.loss(y_hat, enc_labels)
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        # aug_batch = self.augmentation(batch)
        aug_batch = batch["acc_norm"].unsqueeze(dim=1)
        _, loss = self.share_step(aug_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self.share_step(batch)
        logits = torch.argmax(y_hat, dim=1)
        score = self.metric(logits, batch["label"])
        self.log("valid/loss", loss)
        self.log("valid/score", score)
        return loss  # , accu


class CnnModel(pl.LightningModule):
    def __init__(
        self,
        n_class,
        header_in_features: int,
        lr: float = 1e-6,
        weight_decay: float = 1e-3,
        audio_model: Optional[Any] = None,
        accel_model: Optional[Any] = None,
        header: Optional[Any] = None,
    ):
        super().__init__()

        if audio_model:
            self.audio_model = audio_model
        else:
            self.audio_model = AudioCnn1d(fc_layer=False)
        if accel_model:
            self.accel_model = accel_model
        else:
            self.accel_model = AcceleroCnn1d(fc_layer=False)
        if header:
            self.header = header
        else:
            self.header = DnnHeader(n_class=n_class, in_features=header_in_features)

        # TRAINING PARAMETERS
        self.n_class = n_class
        self.lr = lr
        self.wd = weight_decay
        self.decision_layer = nn.Softmax(dim=1)
        self.metric = torchmetrics.F1Score()

    def forward(self, batch):
        audio, accel = batch["audio"], batch["acc_norm"]
        # calculate audio features map
        audio_embs = self.audio_model(audio.unsqueeze(dim=1))
        accel_embs = self.accel_model(accel.unsqueeze(dim=1))
        flatten_map = torch.cat(
            [
                torch.flatten(audio_embs, start_dim=1),
                torch.flatten(accel_embs, start_dim=1),
            ],
            axis=-1,
        )
        embeddings = self.header(flatten_map)
        return self.decision_layer(embeddings)

    def loss(self, y_hat, target):
        return nn.CrossEntropyLoss()(y_hat.float(), target.float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return optimizer

    def share_step(self, batch):
        labels = batch["label"]
        y_hat = self(batch)
        enc_labels = nn.functional.one_hot(labels % self.n_class)
        loss = self.loss(y_hat, enc_labels)
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        # aug_batch = self.augmentation(batch)
        aug_batch = batch
        _, loss = self.share_step(aug_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # self.audio_model.eval()
        # self.accel_model.eval()
        y_hat, loss = self.share_step(batch)
        self.log("valid/loss", loss)  # log loss
        logits = torch.argmax(y_hat, dim=1)
        score = self.metric(logits, batch["label"])
        self.log("valid/score", score)
        return loss
