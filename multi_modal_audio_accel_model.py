# Typing
from typing import List, Dict, Any, Optional, Union

# IO
import os
from pathlib import Path
import json
from datasets import Dataset, DatasetDict

# scientific packages
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.preprocessing import StandardScaler
from quara.ml.utils.results_multi import MultiClassificationResult

# Neural network pytorch
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from quara.ml.classifiers.torch.helpers import FeatTorchDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch import nn
import torchmetrics

# to manage logs
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

# augmentation
from audiomentations import (
    Compose,
    AddGaussianNoise,
    PitchShift,
    Shift,
    Normalize,
    BandStopFilter,
    TimeMask,
)
from augmentation import AddRandomStationaryNoise

# from local
from cnn_model import AudioCnn1d, AcceleroCnn1d, CnnModel


seed_everything(42)

datasets = DatasetDict.load_from_disk("path_to_dict")
datasets.set_format(type="torch", columns=["audio", "acc_norm", "label"])

# AUGMENTATION
n_freq = 3  # number of frequence to add
augmentation = Compose(
    [
        Normalize(p=1),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.3, p=0.5),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
        Shift(
            min_fraction=-0.001,  # min percentage of signal to shift
            max_fraction=0.05,  # max percentage of signal to shift
            rollover=True,
            fade=False,
            fade_duration=0.01,
            p=0.5,
        ),
        *[
            AddRandomStationaryNoise(
                min_frequency=100.0,
                max_frequency=2000.0,
                min_amplitude=0.001,
                max_amplitude=0.3,
                p=0.5,
            )
            for _ in range(n_freq)
        ],
        BandStopFilter(
            min_center_freq=100.0,
            max_center_freq=16000.0,
            min_bandwidth_fraction=0.01,
            max_bandwidth_fraction=0.1,
            min_rolloff=6,
            max_rolloff=12,
            zero_phase=False,
            p=0.5,
        ),
        TimeMask(  # put continuously 5 to 10% of time duration to 0
            min_band_part=0.05, max_band_part=0.1, fade=True, p=0.5
        ),
    ]
)


# Training parameters
BATCH_SIZE = 32
MAX_EPOCHS = 1000

# Prepare DataLoader
train_loader = DataLoader(
    datasets["train"],
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=True,  # shuffle to avoid order biais
)

val_loader = DataLoader(
    datasets["valid"],
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=False,  # avoid to shuffle to validate with same dataset
)


audio_model = AudioCnn1d(n_class=2)
accel_model = AcceleroCnn1d(n_class=2)

plt.plot(next(iter(train_loader))["audio"][0])
plt.show()
audio_feature_map = audio_model(next(iter(train_loader))["audio"].unsqueeze(dim=1))
print("shape of audio featmap", audio_feature_map.shape)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(audio_feature_map[0].detach().numpy())
plt.show()

plt.plot(next(iter(train_loader))["acc_norm"][0])
plt.show()
accel_feature_map = accel_model(next(iter(train_loader))["acc_norm"].unsqueeze(dim=1))
print("shape of audio featmap", accel_feature_map.shape)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(accel_feature_map[0].detach().numpy())
plt.show()

mlp_dim_input = (audio_feature_map.shape[1] * audio_feature_map.shape[2]) + (
    accel_feature_map.shape[1] * accel_feature_map.shape[2]
)
print("MLP input dimensions:", mlp_dim_input)


# PRETRAINED AUDIO
# params to stop the model training and avoid overfitting
audio_logger = TensorBoardLogger(save_dir="logs", name="audio_model")

early_stop_callback = EarlyStopping(
    monitor="valid/loss", min_delta=0.00, patience=200, verbose=False, mode="min"
)

audio_trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,  # should be =1 if you want to monitor the earlystopping on valid/loss,
    callbacks=[early_stop_callback],
    logger=audio_logger,
)

# Init the model
audio_model = AudioCnn1d(n_class=2, lr=1e-6)
# Train the model with the slected parameters
audio_trainer.fit(audio_model, train_loader, val_loader)
# Freeze the models state
audio_model.freeze()

print("Model trained and freezed !")


# PRETRAIN ACCELERO

# TRAIN HEADER

# FINETUNE MODEL
# params to stop the model training and avoid overfitting
early_stop_callback = EarlyStopping(
    monitor="valid/loss", min_delta=0.00, patience=200, verbose=False, mode="min"
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,  # should be =1 if you want to monitor the earlystopping on valid/loss,
    callbacks=[early_stop_callback],
    # logger=logger,
)

# Init the model
model = CnnModel(n_class=2, header_in_features=1480)
# Train the model with the slected parameters
trainer.fit(model, train_loader, val_loader)
# Freeze the models state
model.freeze()

print("Model trained and freezed !")
