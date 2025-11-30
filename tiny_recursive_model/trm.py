from __future__ import annotations
from contextlib import nullcontext

import torch
from torch import nn, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce, pack, unpack

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper
        halt_loss_weight = 1.,
        num_register_tokens = 0,
        max_seq_len = 1024
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.input_embed = nn.Embedding(num_tokens, dim)
        
        # 1. Positional Embeddings (Crucial for Logic)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        # 2. State Normalization (Crucial for Depth)
        self.latent_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)

        self.latent_gate = nn.Linear(dim, dim)
        self.output_gate = nn.Linear(dim, dim)

        # --- FIX: NEUTRAL INITIALIZATION (0.0) ---
        # Bias 0.0 -> Sigmoid(0) = 0.5.
        # This makes the gate "Neutral". It allows the model to choose 
        # freely between "Remembering" (Recall) and "Flipping" (Parity).
        nn.init.constant_(self.latent_gate.bias, 0.0)
        nn.init.constant_(self.output_gate.bias, 0.0)

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        self.to_pred = nn.Linear(dim, num_tokens, bias = False)

        self.to_halt_pred = nn.Sequential(
            nn.Linear(dim, 1, bias = False),
            nn.Sigmoid()
        )

        self.halt_loss_weight = halt_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed
        return outputs, latents

    def embed_inputs_with_registers(self, seq):
        batch, seq_len = seq.shape
        inputs = self.input_embed(seq)
        positions = arange(seq_len, device=self.device)
        inputs = inputs + self.pos_embed(positions)
        registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        inputs, packed_shape = pack([registers, inputs], 'b * d')
        return inputs, packed_shape

    def refine_latent_then_output_once(self, inputs, outputs, latents):
        # Latent loop
        for _ in range(self.num_latent_refinements):
            combined = outputs.add(latents).add(inputs)
            normed_combined = self.latent_norm(combined)
            
            # --- FIX: GATED UPDATE ---
            # 1. Calculate Candidate Update
            update = self.network(normed_combined)
            # 2. Calculate Gate (0 = Locked, 1 = Open)
            gate = torch.sigmoid(self.latent_gate(normed_combined))
            # 3. Smoothly Blend
            latents = (1. - gate) * latents + gate * update

        # Output refinement
        combined_out = outputs.add(latents)
        normed_combined_out = self.output_norm(combined_out)
        
        # --- FIX: GATED UPDATE ---
        out_update = self.network(normed_combined_out)
        out_gate = torch.sigmoid(self.output_gate(normed_combined_out))
        outputs = (1. - out_gate) * outputs + out_gate * out_update

        return outputs, latents

    def deep_refinement(self, inputs, outputs, latents):
        for step in range_from_one(self.num_refinement_blocks):
            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

        return outputs, latents

    @torch.no_grad()
    def predict(
        self,
        seq,
        halt_prob_thres = 0.5,
        max_deep_refinement_steps = 12
    ):
        batch = seq.shape[0]
        inputs, packed_shape = self.embed_inputs_with_registers(seq)
        outputs, latents = self.get_initial()

        active_batch_indices = arange(batch, device = self.device, dtype = torch.float32)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps

            outputs, latents = self.deep_refinement(inputs, outputs, latents)

            halt_prob = self.to_halt_pred(outputs.mean(dim=1)).squeeze(-1)

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')
            
            pred = self.to_pred(outputs_for_pred[should_halt])
            preds.append(pred)

            count = should_halt.sum().item()
            exited_step_indices.extend([step] * count)
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            not_halt = ~should_halt
            inputs = inputs[not_halt]
            outputs = outputs[not_halt]
            latents = latents[not_halt]
            active_batch_indices = active_batch_indices[not_halt]

            if is_empty(outputs):
                break

        preds = cat(preds).argmax(dim = -1)
        exited_step_indices = tensor(exited_step_indices, device=self.device)
        exited_batch_indices = cat(exited_batch_indices)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        seq,
        outputs,
        latents,
        labels = None
    ):
        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        outputs, latents = self.deep_refinement(inputs, outputs, latents)

        registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

        pred = self.to_pred(outputs_for_pred)
        halt_prob = self.to_halt_pred(outputs.mean(dim=1)).squeeze(-1)

        return_package = (outputs, latents, pred, halt_prob)

        if not exists(labels):
            return return_package

        loss = F.cross_entropy(pred.transpose(1, 2), labels, reduction = 'none')
        
        valid_token_count = (labels != -100).sum(dim=-1).float()
        valid_token_count = valid_token_count.clamp(min=1.0)
        
        loss = loss.sum(dim=-1) / valid_token_count

        is_all_correct = (pred.argmax(dim = -1) == labels).all(dim = -1)
        halt_loss = F.binary_cross_entropy(halt_prob, is_all_correct.float(), reduction = 'none')

        total_loss = (loss + halt_loss * self.halt_loss_weight)
        losses = (loss, halt_loss)

        return (total_loss.sum(), losses, *return_package)