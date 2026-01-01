## Environment and imports

This module sets up the runtime and imports everything needed for a SimCLR-style self-supervised pretraining run using PyTorch + Lightning + TorchVision.

Key dependencies used:
- `torch`, `torchvision`, `torch.nn`, `torch.nn.functional`: model definition, feature extraction, tensor ops, cosine similarity, etc.
- `lightning` (PyTorch Lightning): training loop orchestration, logging, deterministic seeding.
- `torchmetrics`: accuracy metrics used as proxy signals during self-supervised training.
- `CSVLogger`: writes epoch metrics to `metrics.csv` for later plotting.
- Local utilities imported from `sharedutilities` (not defined inside this file): `LightningModel`, `Cifar10DataModule`, and `plotlossandacc` are referenced for data and plotting in earlier cells.

Reproducibility is enforced via `L.pytorch.seed_everything(123)`, and the CIFAR-10 dataset is prepared and set up through the data module.

## Data module (CIFAR-10)

This module uses a Lightning DataModule abstraction (named `Cifar10DataModule`) to provide train/val/test dataloaders for CIFAR-10.

Important behavior for SimCLR:
- The training dataloader returns images in a way that supports producing **two augmented views** per original image (either by returning a tuple/list of views or by applying a transform wrapper that yields two views).
- Batch size and workers are configurable (examples shown: `batchsize=64` and later `batchsize=256`, `numworkers=4`).

## Self-supervised augmentations (SimCLR-style)

This module defines the transformation pipeline used to create two correlated views of the same image (the core SimCLR idea).

Main transform pipeline (`selfsupervised_transforms`):
- `RandomResizedCrop(size=128)` to change scale/position.
- `RandomHorizontalFlip`.
- `RandomApply(ColorJitter(...), p=0.8)` to perturb color strongly.
- `RandomGrayscale(p=0.2)`.
- `GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))`.
- `ToTensor`.

### `AugmentedImages` wrapper

`AugmentedImages` is a small callable wrapper whose `__call__` returns **two independently augmented tensors** from the same input image by applying the same transform pipeline twice.

This is how the pipeline produces the “(view1, view2)” pair required by contrastive learning without needing labels.

## Backbone encoder model (ResNet)

This module uses a ResNet-based backbone (the notebook later saves weights into a file named like `simclr-resnet18.pt`).

Conceptually, this model is used as an **encoder** that maps an input image tensor to a feature vector (embedding), which is then used in the contrastive loss.

## Contrastive objective: `infonce_loss`

This module implements the InfoNCE loss used by SimCLR.

Core steps performed in `infonce_loss(feats, mode, temperature)`:
- Compute pairwise cosine similarity between all feature vectors in the batch via `F.cosine_similarity`.
- Mask out self-similarities (diagonal) so a sample is not compared to itself.
- Identify the “positive pair” for each sample as the corresponding augmented view located `batch_size` positions away after concatenation (the classic SimCLR batching trick: stack view1 and view2).
- Divide similarities by `temperature` and compute a cross-entropy-like negative log-likelihood using `logsumexp` for numerical stability.
- Return:
  - `nll` (the InfoNCE loss scalar)
  - `sim_argsort`, a ranking signal derived from similarities (used downstream as a proxy “accuracy”).

The temperature hyperparameter (e.g., `0.07`) controls how “peaky” the softmax distribution over similarities becomes.

## Lightning training module: `LightningModelSimCLR`

This module defines the LightningModule that plugs the encoder + InfoNCE loss into a full training loop.

### Initialization

`LightningModelSimCLR(model, learningrate, temperature)` stores:
- `self.model`: the encoder backbone
- `self.learningrate`, `self.temperature`: optimization and loss hyperparameters
- `torchmetrics.Accuracy(task="multiclass", num_classes=10)` for train/val/test (used as a proxy metric, not true classification accuracy).

### Forward pass

`forward(x)` simply delegates to the underlying encoder: `return self.model(x)`.

### Training step

In `training_step`:
- The batch provides `images, true_labels`, but labels are unused for the self-supervised objective.
- `images` contains two augmented views; they are concatenated along the batch dimension via `torch.cat(images, dim=0)`.
- The encoder produces embeddings: `transformed_feats = self(images)`.
- `infonce_loss` computes loss and ranking statistics.
- It logs:
  - `train_loss`
  - a proxy `train_acc` computed from the similarity ranking (e.g., whether the positive is in the top-k, such as `sim_argsort < 5`).

### Validation step

Validation mirrors the training step but logs `val_loss` and `val_acc` from the same InfoNCE-based ranking idea.

### Optimizer

`configure_optimizers` uses Adam with the specified learning rate.

## Training orchestration and logging

This module configures and runs the Lightning trainer.

Key pieces:
- `CSVLogger(savedir="logs", name="my-model")` writes metrics to a versioned folder that includes a `metrics.csv`.
- `Trainer(max_epochs=50, accelerator="gpu", devices=1, deterministic=True, logger=...)` runs the full training loop with deterministic behavior enabled.
- A larger batch size (example shown: 256) is used for contrastive learning, which often benefits from more negatives per batch.

## Metrics plotting utility

This module includes a `plotlossandacc(logdir, ...)` function that reads the CSV logs, aggregates by epoch, and plots train/val loss and train/val “accuracy” curves.

It is intended for quick inspection of training dynamics rather than being a core part of the SimCLR method itself.

## Saving pretrained weights

This final module saves the encoder weights to disk using `torch.save(model.state_dict(), "simclr-resnet18.pt")`.

This artifact is the result of self-supervised pretraining and is typically reused later for:
- Linear probing (training a classifier head on frozen features)
- Fine-tuning the full network on labeled data.

## Goal

This code’s goal is to **pretrain an image encoder using SimCLR-style self-supervised learning** on CIFAR-10: it learns good feature embeddings from *unlabeled* images by making two augmented views of the same image map close together in embedding space, and views from different images map far apart.

### What it implements
- It builds a ResNet-18 backbone and replaces the classification head with a small MLP “projection head” that outputs a 256‑dim embedding (not class logits).
- It creates two strong augmentations of each image, concatenates them into one batch, and uses an InfoNCE contrastive loss to train the network to match each image to its paired augmentation among many negatives in the batch.

### How training is done
- The `info_nce_loss(...)` function computes cosine similarities, masks self-similarity, identifies the positive pair by a batch shift, applies temperature scaling, and computes the contrastive (NCE) loss via `logsumexp`.
- A LightningModule (`LightningModelSimCLR`) wraps the model and trains/validates by logging the InfoNCE loss plus a proxy “accuracy” based on how highly the true positive ranks among similarities.

### What you get out of it
- The output is a pretrained representation model (encoder + projection head) whose embeddings can later be used for downstream tasks (e.g., linear probing / fine-tuning for classification) with fewer labels.
- The notebook also visualizes an “augmented batch” to show the two views being contrasted.

## Future work
You’d use this SimCLR-pretrained model later as a **feature extractor / initialization** for a labeled task (or even an unlabeled retrieval/clustering task), instead of training a network from scratch.

### How to use it later
- **Load the pretrained weights**: the notebook saves the trained state dict with `torch.save(pytorchmodel.state_dict(), "simclr-resnet18.pt")`.
- **Use the encoder as a frozen feature extractor (“linear probe”)**: keep the pretrained ResNet weights fixed, replace the projection head with a linear classifier (e.g., `Linear(512, num_classes)`), and train only that last layer on labeled data. This tests the quality of the learned representation.
- **Fine-tune for your downstream task**: initialize from these weights, then train the whole network (or unfreeze progressively) on your labeled dataset; typically you drop the SimCLR projection head (`pytorchmodel.fc` is an MLP producing 256-d embeddings) and attach a task head instead.
- **Use embeddings directly** (no classifier): keep the model in “embedding mode” and run images through it to get 256‑d vectors; then do kNN retrieval, clustering, anomaly detection, or train a small classical model on top of embeddings.

### Benefits (why it’s worth it)
- **Less labeled data needed**: self-supervised pretraining learns general-purpose visual features using only augmentations and the InfoNCE objective, so downstream performance can be strong even with limited labels.
- **Faster convergence and often better accuracy**: starting from a representation already shaped by contrastive learning usually trains faster than random init and can generalize better, especially when the downstream dataset is small or distribution-shifted.
- **Reusable representation**: the same pretrained encoder can be reused across multiple tasks/datasets; you only swap/fit the head (linear probe) or lightly fine-tune.
