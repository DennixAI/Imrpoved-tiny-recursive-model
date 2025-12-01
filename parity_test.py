import torch
from torch.utils.data import Dataset, DataLoader
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

# --- 1. Define the Parity Dataset ---
# Logic Task: Cumulative Sum Modulo 2
# Input: [1, 0, 1, 1] -> Target: [1, 1, 0, 1]
class ParityDataset(Dataset):
    def __init__(self, samples=2000, seq_len=32):
        self.seq_len = seq_len
        self.data = torch.randint(0, 2, (samples, seq_len))
        self.targets = self.data.cumsum(dim=1) % 2 

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

def run_test():
    print("--- TESTING ORIGINAL ARCHITECTURE ON PARITY ---")
    
    # Settings derived from original repo defaults
    SEQ_LEN = 32
    DIM = 256
    
    # 2. Setup Original Model
    # Note: Original TRM does NOT have 'max_seq_len' or 'pos_embed' logic in __init__
    # It relies on MLPMixer's dense weights for position awareness.
    model = TinyRecursiveModel(
        num_tokens = 2,          # Binary inputs (0, 1)
        dim = DIM,
        num_refinement_blocks = 2,
        num_latent_refinements = 4,
        network = MLPMixer1D(
            dim = DIM, 
            depth = 2,
            seq_len = SEQ_LEN
        )
    )

    # 3. Setup Original Trainer
    # Note: We use standard settings. The original trainer performs 
    # optimizer.step() INSIDE the recurrence loop (Broken BPTT).
    dataset = ParityDataset(samples=2000, seq_len=SEQ_LEN)
    
    trainer = Trainer(
        model,
        dataset,
        batch_size = 16,
        epochs = 5,
        max_recurrent_steps = 12,
        learning_rate = 1e-4,
        cpu = not torch.cuda.is_available()
    )

    # 4. Train
    print("Training starting...")
    trainer.forward() # This runs the training loop
    
    # 5. Validation
    print("\nValidating...")
    device = trainer.accelerator.device
    
    # Generate a fresh batch
    val_inputs = torch.randint(0, 2, (16, SEQ_LEN)).to(device)
    val_targets = val_inputs.cumsum(dim=1) % 2
    
    # Predict
    model.eval()
    preds, _ = model.predict(val_inputs)
    
    # Check Accuracy
    correct = (preds == val_targets).sum().item()
    total = val_inputs.numel()
    acc = (correct / total) * 100
    
    print(f"Final Accuracy: {acc:.2f}%")
    
    # Check simple logic: If acc is ~50%, it failed (Random Guessing).
    if acc > 95:
        print("RESULT: PASSED. The original code can solve Parity.")
    elif acc < 60:
        print("RESULT: FAILED. The original code is guessing randomly.")
    else:
        print("RESULT: PARTIAL. It learned something but is unstable.")

if __name__ == "__main__":
    run_test()
