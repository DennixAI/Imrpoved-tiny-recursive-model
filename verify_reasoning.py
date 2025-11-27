import torch
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- DIAGNOSTIC SUITE: REASONING & MEMORY ---")

# --- TASK 1: PARITY CHECK (Logic) ---
class ParityDataset(Dataset):
    def __init__(self, samples=2000, seq_len=64):
        self.seq_len = seq_len
        self.data = torch.randint(0, 2, (samples, seq_len))
        self.targets = self.data.sum(dim=1) % 2 
        self.targets = self.targets.unsqueeze(1).repeat(1, seq_len)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

# --- TASK 2: ASSOCIATIVE RECALL (Memory) ---
class AssociativeRecallDataset(Dataset):
    def __init__(self, samples=2000, seq_len=32):
        self.seq_len = seq_len # Length of the Key-Value pairs only
        self.samples = samples
    
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        # Format: Key (0-9), Value (10-19)... Query Key
        keys = torch.randint(0, 10, (self.seq_len // 2,))
        values = keys + 10 
        
        seq = torch.stack((keys, values), dim=1).flatten()
        
        query_idx = torch.randint(0, len(keys), (1,))
        query_key = keys[query_idx]
        target_val = values[query_idx]
        
        inp = torch.cat((seq, query_key)) # Length is seq_len + 1
        
        # Target: We only care about the last prediction
        tgt = torch.full_like(inp, target_val.item())
        
        return inp, tgt

# --- RUNNER ---
def run_task(name, dataset, epochs=1):
    print(f"\n[TEST] Starting Task: {name}")
    
    # Get actual sequence length from a sample to ensure model matches data
    sample_inp, _ = dataset[0]
    actual_seq_len = sample_inp.shape[0]
    print(f"      Detected Sequence Length: {actual_seq_len}")
    
    # 1. Tiny Model Config
    model = TinyRecursiveModel(
        dim = 64,
        num_tokens = 256,
        network = MLPMixer1D(
            dim = 64, 
            depth = 2, 
            seq_len = actual_seq_len # <--- FIXED: Dynamic Size
        ),
        num_refinement_blocks = 2,
        num_latent_refinements = 4 
    )

    # 2. Trainer
    trainer = Trainer(
        model,
        dataset,
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 8,
        compile_model = True,
        cpu = not torch.cuda.is_available()
    )

    # 3. Train
    trainer.forward()
    
    # 4. Quick Sanity Check (Inference)
    print(f"      Validating {name}...")
    inp, tgt = dataset[0]
    inp = inp.unsqueeze(0).to(trainer.accelerator.device)
    pred, _ = model.predict(inp)
    
    print(f"      Prediction Sample (Last Token): {pred[0, -1].item()}")
    print(f"      Target Sample (Last Token):     {tgt[-1].item()}")
    
    if pred[0, -1].item() == tgt[-1].item():
        print(f"      >>> SUCCESS: {name} Passed.")
    else:
        print(f"      >>> FAILURE: {name} Failed.")

if __name__ == "__main__":
    # 1. Logic Test
    run_task("Parity Check", ParityDataset(samples=1000, seq_len=64), epochs=2)
    
    # 2. Memory Test
    # Note: Associative Recall has input length 33 (32 pairs + 1 query)
    run_task("Associative Recall", AssociativeRecallDataset(samples=1000, seq_len=32), epochs=4)