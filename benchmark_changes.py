import torch
import time
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

# 1. Setup a Dummy Dataset (Fast)
class RandomDataset(Dataset):
    def __init__(self, length=100, seq_len=128):
        self.length = length
        self.seq_len = seq_len
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # Random inputs and targets
        return torch.randint(0, 256, (self.seq_len,)), torch.randint(0, 256, (self.seq_len,))

def run_benchmark():
    print("--- Starting Benchmark of New Optimizations ---")
    
    # Check Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == 'cpu':
        print("WARNING: torch.compile and fused ops are much slower on CPU. Use GPU for true speed test.")

    # 2. Initialize Model (Small config for speed)
    model = TinyRecursiveModel(
        dim = 128,              # Small dim
        num_tokens = 256,
        network = MLPMixer1D(   # Using your new Optimized Mixer
            dim = 128,
            depth = 2,
            seq_len = 128
        ),
        num_refinement_blocks = 2,
        num_latent_refinements = 2
    )

    # 3. Initialize Trainer with NEW Flags
    trainer = Trainer(
        model,
        RandomDataset(length=64, seq_len=128), # Small dataset
        batch_size = 16,
        epochs = 1,
        max_recurrent_steps = 4, # Short loop for test
        compile_model = True,    # <--- TESTING YOUR COMPILE UPGRADE
        cpu = (device == 'cpu')
    )

    print("\n[1/3] Starting Training Loop...")
    print("      (Note: Step 1 will lag while torch.compile fuses kernels...)")
    
    start_time = time.time()
    
    # Run the Forward pass (Training)
    trainer.forward()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n[2/3] Training Complete.")
    print(f"      Total Time: {total_time:.4f}s")
    
    # 4. Verify Inference (Did we break predict?)
    print("\n[3/3] Testing Inference (Predict)...")
    dummy_input = torch.randint(0, 256, (1, 128)).to(device)
    
    # This triggers the 'predict' method which we also touched
    pred, steps = trainer.model.predict(dummy_input)
    
    print(f"      Prediction Output Shape: {pred.shape}")
    print(f"      Steps taken: {steps}")
    print("\n--- SUCCESS: Changes are working! ---")

if __name__ == "__main__":
    run_benchmark()