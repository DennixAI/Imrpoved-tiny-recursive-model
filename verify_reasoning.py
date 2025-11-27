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
        self.seq_len = seq_len
        self.samples = samples
    
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        keys = torch.randint(0, 10, (self.seq_len // 2,))
        values = keys + 10 
        seq = torch.stack((keys, values), dim=1).flatten()
        query_idx = torch.randint(0, len(keys), (1,))
        query_key = keys[query_idx]
        target_val = values[query_idx]
        inp = torch.cat((seq, query_key))
        tgt = torch.full_like(inp, target_val.item())
        return inp, tgt

def run_task(name, dataset, epochs=5, num_tokens=256):
    print(f"\n[TEST] Starting Task: {name}")
    
    sample_inp, _ = dataset[0]
    actual_seq_len = sample_inp.shape[0]
    print(f"      Detected Sequence Length: {actual_seq_len}")
    
    # 1. Tiny Model Config
    model = TinyRecursiveModel(
        dim = 64,
        num_tokens = num_tokens, # Configurable vocab size
        network = MLPMixer1D(
            dim = 64, 
            depth = 2, 
            seq_len = actual_seq_len 
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
    
    final_pred = pred[0, -1].item()
    final_tgt = tgt[-1].item()
    
    print(f"      Prediction: {final_pred}")
    print(f"      Target:     {final_tgt}")
    
    if final_pred == final_tgt:
        print(f"      >>> SUCCESS: {name} Passed.")
    else:
        print(f"      >>> FAILURE: {name} Failed.")

if __name__ == "__main__":
    # Task 1: Parity (Easier Config: 16 tokens, 10 epochs)
    run_task("Parity Check", ParityDataset(samples=2000, seq_len=64), epochs=10, num_tokens=16)
    
    # Task 2: Associative Recall
    run_task("Associative Recall", AssociativeRecallDataset(samples=2000, seq_len=32), epochs=5, num_tokens=256)