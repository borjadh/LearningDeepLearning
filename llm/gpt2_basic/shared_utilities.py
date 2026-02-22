import math
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate=3e-4, warmup_steps=2000):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {self.num_params/1e6:.2f}M parameters")
    
    def forward(self, idx, targets=None):
        return self.model(idx, targets)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, loss = self(inputs, targets)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, loss = self(inputs, targets)
        
        perplexity = torch.exp(loss)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, loss = self(inputs, targets)
        
        perplexity = torch.exp(loss)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_perplexity", perplexity, prog_bar=True)
        
        return loss

    
    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Learning rate scheduler: warmup + cosine decay
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.trainer.max_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    """
    def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Use a simple warmup scheduler that doesn't need max_steps
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=self.warmup_steps
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
    """
        

class HFTextDataModule(L.LightningDataModule):
    """Modern DataModule using HuggingFace Datasets."""
    
    def __init__(
        self,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        batch_size=16,
        block_size=256,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
    def prepare_data(self):
        """Download dataset (runs only once on main process)."""
        load_dataset(self.dataset_name, self.dataset_config)
        GPT2TokenizerFast.from_pretrained("gpt2")
    
    def setup(self, stage=None):
        """Load and tokenize dataset."""
        # Load tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config)
        
        # Tokenize each split separately
        print("Tokenizing dataset...")
        train_tokens = self._tokenize_split(dataset["train"])
        val_tokens = self._tokenize_split(dataset["validation"])
        self.test_tokens = self._tokenize_split(dataset["test"])
        
        # Create PyTorch datasets
        self.train = TextDataset(train_tokens, self.block_size)
        self.val = TextDataset(val_tokens, self.block_size)
        
        print(f"\n✅ Dataset loaded:")
        print(f"   Vocab size: {len(self.tokenizer):,}")
        print(f"   Train tokens: {len(train_tokens):,}")
        print(f"   Val tokens: {len(val_tokens):,}")
        print(f"   Test tokens: {len(self.test_tokens):,}")
        print(f"   Train samples: {len(self.train):,}")
        print(f"   Val samples: {len(self.val):,}")
    
    def _tokenize_split(self, split_dataset):
        """Tokenize a dataset split and return flattened tokens."""
        all_tokens = []
        
        for example in split_dataset:
            text = example["text"]
            # Skip empty lines
            if text.strip():
                # Tokenize without truncation
                tokens = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    truncation=False,
                )["input_ids"]
                all_tokens.extend(tokens)
        
        return torch.tensor(all_tokens, dtype=torch.long)
    
    def train_dataloader(self):
        print("Creating train dataloader...")
        loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        print(f"Train dataloader created with {len(loader)} batches")
        return loader
    
    def val_dataloader(self):
        print("Creating val dataloader...")
        loader = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        print(f"Val dataloader created with {len(loader)} batches")
        return loader

class TextDataset(Dataset):
    """Simple dataset for language modeling."""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class TextDataset(Dataset):
    """Simple dataset for language modeling."""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y