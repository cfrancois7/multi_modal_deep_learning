from torch import nn
import pytorch_lightning as pl


class DnnHeader(pl.LightningModule):
    def __init__(self, in_features: int, n_class: int):
        super().__init__()
        dropout = 0.2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),  # TODO to define the input with formula
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm2d(64)
            nn.Linear(32, n_class),
        )

    def forward(self, x):
        return self.mlp(x)
