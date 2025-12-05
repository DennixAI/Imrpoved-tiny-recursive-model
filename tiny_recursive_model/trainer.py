from __future__ import annotations

import os
import re
import torch
from pathlib import Path
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from einops import pack, unpack

from accelerate import Accelerator
from ema_pytorch import EMA

from tiny_recursive_model.trm import TinyRecursiveModel

try:
    from adam_atan2_pytorch import MuonAdamAtan2
except ImportError:
    MuonAdamAtan2 = None

from x_transformers import Encoder, Decoder

def exists(v):
    return v is not None

def range_from_one(n):
    return range(1, n + 1)

def is_empty(t):
    return t.numel() == 0

class Trainer(Module):
    def __init__(
        self,
        model: TinyRecursiveModel | Module,
        dataset: Dataset,
        optim_klass = AdamW,
        optim: Optimizer | None = None,
        learning_rate = 1e-4,
        muon_learning_rate = 1e-3,
        weight_decay = 1.,
        batch_size = 16,
        epochs = 2,
        halt_prob_thres = 0.5,
        max_recurrent_steps = 12,
        warmup_steps = 2000,
        ema_decay_rate = 0.999,
        switch_ema_every = 10000,
        accelerate_kwargs: dict = dict(),
        cpu = False,
        compile_model = True,
        # --- NEW ARGUMENT: Checkpoint Directory ---
        checkpoint_folder = './checkpoints' 
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs, cpu=cpu)
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset
        
        # Checkpointing setup
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4 if not cpu else 0,
            pin_memory = True if not cpu else False,
            prefetch_factor = 2 if not cpu else None
        )

        if not exists(optim):
            if isinstance(model.network, (Encoder, Decoder)) and MuonAdamAtan2 is not None:
                optim = MuonAdamAtan2(
                    model.network.muon_parameters(),
                    model.parameters(),
                    lr = learning_rate / batch_size,
                    muon_lr = muon_learning_rate / batch_size,
                    weight_decay = weight_decay
                )
            else:
                optim = optim_klass(
                    model.parameters(),
                    lr = learning_rate / batch_size,
                    weight_decay = weight_decay
                )

        self.optim = optim
        self.scheduler = LambdaLR(self.optim, lambda step: min((step + 1) / warmup_steps, 1.0))
        self.model = model
        self.ema_model = None

        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta = ema_decay_rate,
                update_model_with_ema_every = switch_ema_every,
                forward_method_names = ('predict',)
            )
            self.ema_model.to(self.accelerator.device)

        self.halt_prob_thres = halt_prob_thres
        self.max_recurrent_steps = max_recurrent_steps

        if compile_model and not cpu and hasattr(torch, "compile"):
            self.accelerator.print("Compiling model with torch.compile (mode='default')...")
            self.model = torch.compile(self.model, mode="default")

        self.model, self.optim, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optim, self.dataloader, self.scheduler
        )

    def save_checkpoint(self, epoch):
        # Saves to ./checkpoints/epoch-N
        save_path = self.checkpoint_folder / f"epoch-{epoch}"
        self.accelerator.save_state(save_path)
        self.accelerator.print(f"Checkpoint saved: {save_path}")

    def load_last_checkpoint(self):
        # find the highest epoch number in the folder
        if not self.checkpoint_folder.exists():
            return 1 # Start at epoch 1

        folders = [f for f in os.listdir(self.checkpoint_folder) if f.startswith("epoch-")]
        if not folders:
            return 1
        
        # Extract numbers from "epoch-10", "epoch-5"
        epochs_found = [int(re.findall(r'\d+', f)[0]) for f in folders]
        last_epoch = max(epochs_found)
        
        load_path = self.checkpoint_folder / f"epoch-{last_epoch}"
        self.accelerator.print(f"Resuming from checkpoint: {load_path}")
        self.accelerator.load_state(load_path)
        
        return last_epoch + 1

    def forward(self):
        loss_history = []
        

        start_epoch = self.load_last_checkpoint()
        if start_epoch > self.epochs:
            self.accelerator.print(f"Training already completed ({start_epoch-1}/{self.epochs} epochs).")
            return loss_history

        for epoch in range(start_epoch, self.epochs + 1):
            for dataset_input, dataset_output in self.dataloader:
                
                self.optim.zero_grad()
                outputs, latents = self.model.get_initial()
                
                total_loss_accumulated = 0
                
                for recurrent_step in range_from_one(self.max_recurrent_steps):
                    
                    loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(
                        dataset_input, outputs, latents, labels = dataset_output
                    )

                    total_loss_accumulated += (loss / self.max_recurrent_steps)

                    halt_mask = halt >= self.halt_prob_thres
                    if not halt_mask.any(): continue
                    
                    if halt_mask.any():
                        outputs = outputs[~halt_mask]
                        latents = latents[~halt_mask]
                        dataset_input = dataset_input[~halt_mask]
                        dataset_output = dataset_output[~halt_mask]

                    if is_empty(outputs): break
                
                current_loss = main_loss.mean().item()
                loss_history.append(current_loss)
                self.accelerator.print(f'[{epoch}] loss: {current_loss:.3f}')
                
                self.accelerator.backward(total_loss_accumulated)
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optim.step()
                self.scheduler.step()

                if self.accelerator.is_main_process:
                    self.ema_model.update()
            
            # save checkpoint
            self.save_checkpoint(epoch)

        self.accelerator.print('complete')

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()
            
        return loss_history