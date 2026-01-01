# Notebook goal (what this does)
This notebook loads a pretrained **ResNet-18** (ImageNet weights), freezes all pretrained parameters, replaces the final classification layer (`fc`) to output 10 classes (CIFAR-10), and trains only that last layer using PyTorch Lightning.

## Cell 1 — Environment/version stamp
```python
%load_ext watermark
%watermark -p torch,lightning,torchvision
```
- Loads the `watermark` Jupyter extension and prints the versions of `torch`, `lightning`, and `torchvision` used to run the notebook (useful for reproducibility).

## Cell 2 — Imports
```python
import lightning as L
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import numpy as np

from shared_utilities import LightningModel, Cifar10DataModule, plot_loss_and_acc
```
- Imports core libraries: PyTorch, TorchVision, Lightning, metrics, plotting utilities.
- `CSVLogger` is used to save training logs to disk in CSV form.
- `LightningModel`, `Cifar10DataModule`, `plot_loss_and_acc` are custom helpers (defined in `shared_utilities`) that wrap the training loop, data loading, and plotting.

## Cell 3 — Discover Torch Hub entrypoints
```python
entrypoints = torch.hub.list('pytorch/vision:v0.13.0', force_reload=True)
for e in entrypoints:
    if "resnet" in e:
        print(e)
```
- Uses Torch Hub to list all models available in the `pytorch/vision:v0.13.0` repo snapshot.
- Filters and prints only the entries containing `"resnet"` so you can see which ResNet variants are available.

## Cell 4 — Load pretrained ResNet-18
```python
pytorch_model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', weights='IMAGENET1K_V1')
```
- Downloads (or loads from cache) a pretrained **ResNet-18** with ImageNet-1K weights (`IMAGENET1K_V1`).
- Result: `pytorch_model` is a ready-to-use TorchVision ResNet object.

## Cell 5 — Alternative loading route (commented)
```python
# Alternatively:
# from torchvision.models import resnet18, ResNet18_Weights
# weights = ResNet18_Weights.IMAGENET1K_V1
```
- Shows the more “direct” TorchVision API alternative to Torch Hub.
- It’s commented out, so it does nothing at runtime.

## Cell 6 — Inspect the model architecture
```python
pytorch_model
```
- Printing the model shows the full module tree: stem (`conv1`, `bn1`, `relu`, `maxpool`), residual stages (`layer1..layer4`), pooling, and the classifier head (`fc`).
- You can confirm that the pretrained head is `Linear(in_features=512, out_features=1000)` (ImageNet classes).

## Cell 7 — Freeze all weights and replace only the last layer
```python
for param in pytorch_model.parameters():
    param.requires_grad = False

pytorch_model.fc = torch.nn.Linear(512, 10)
```
- Loop over every parameter tensor in the pretrained model and sets `requires_grad = False`, which prevents gradient computation and weight updates for those layers.
- Replaces the original ImageNet classifier (`fc`) with a new linear layer mapping from 512 features to 10 outputs (CIFAR-10 classes).
- Important consequence: the new `pytorch_model.fc` parameters default to `requires_grad=True`, so **only the new head is trainable**.

## Cell 8 — Get the pretrained preprocessing transform
```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
preprocess_transform = weights.transforms()
preprocess_transform
```
- Retrieves the standard TorchVision preprocessing pipeline associated with the chosen ImageNet weights.
- The printed transform includes: resize to 256, center crop to 224, and normalize with ImageNet mean/std.
- This matters because pretrained models typically assume the same normalization protocol used during pretraining.

## Cell 9 — Reproducibility + build DataModule, LightningModule, Trainer
```python
%%capture --no-display

L.pytorch.seed_everything(123)

dm = Cifar10DataModule(
    batch_size=64,
    num_workers=4,
    train_transform=preprocess_transform,
    test_transform=preprocess_transform
)

lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=,
    logger=CSVLogger(save_dir="logs/", name="my-model"),
    deterministic=True,
)
```
- `%%capture --no-display` suppresses normal output (cleaner notebook).
- `L.pytorch.seed_everything(123)` sets random seeds for reproducibility across runs.
- `Cifar10DataModule(...)` constructs a Lightning DataModule for CIFAR-10 with batch size 64 and 4 worker processes, applying the ImageNet preprocessing transform to train/test images.
- `LightningModel(model=pytorch_model, learning_rate=0.1)` wraps the PyTorch model into a LightningModule that defines `training_step`, `validation_step`, loss, metrics, and optimizer config (implementation is inside `shared_utilities`).
- `L.Trainer(...)` configures training:
  - `max_epochs=50` trains for 50 epochs.
  - `accelerator="gpu"` uses CUDA.
  - `devices=[2]` uses GPU index 2 specifically (not “2 GPUs”).
  - `CSVLogger(...)` writes metrics to `logs/my-model/...`.
  - `deterministic=True` requests deterministic behavior when possible.

## Cell 10 — Train
```python
trainer.fit(model=lightning_model, datamodule=dm)
```
- Runs the Lightning training loop using the dataloaders from `dm` and the steps/optimizer from `lightning_model`.
- Because you froze all parameters except `fc`, the Trainer summary shows only a small number of **trainable params** (the classifier head) while the rest are non-trainable.

## Cell 11 — Plot training curves
```python
plot_loss_and_acc(trainer.logger.log_dir, loss_ylim=(0.0, 2.0))
```
- Reads the CSV logs produced by `CSVLogger` and plots loss/accuracy curves.
- `loss_ylim=(0.0, 2.0)` sets the y-axis bounds for the loss plot to improve readability.

## Cell 12 — Test evaluation
```python
trainer.test(model=lightning_model, datamodule=dm)
```
- Runs the test loop on the CIFAR-10 test set and prints test metrics (here, `test_acc`).
- The output shown in the notebook reports a `test_acc` around `0.7554` for that run.

## What to document in `shared_utilities` (important dependency)
This notebook relies on `shared_utilities.LightningModel` and `shared_utilities.Cifar10DataModule`, so the “true” training logic (loss function, accuracy computation, optimizer selection, dataloaders, etc.) lives there.

If you want, attach `shared_utilities.py` too and an explanation can be written with the same line-by-line style.
```

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/75896238/91805cd4-0262-40cb-8765-fca742bbac70/7.6-part-3-resnet18-transfer-last-layer-only.ipynb)