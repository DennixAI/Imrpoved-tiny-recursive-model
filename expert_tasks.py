import torch
import random
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- EXPERT SUITE: SET COMPLETION & ARITHMETIC ---")

# --- TASK 1: SET COMPLETION (Sudoku-Lite) ---
# Input: A set of numbers [1, 5, 2, 0, 4] (0 is missing)
# Target: The missing number [3]
class SetCompletionDataset(Dataset):
    def __init__(self, samples=5000, set_size=16):
        self.data = []
        self.set_size = set_size
        
        for _ in range(samples):
            # Generate a full permutation [1, 2, ... N]
            full_set = list(range(1, set_size + 1))
            random.shuffle(full_set)
            
            # Hide one number (set it to 0)
            missing_val = full_set[0]
            full_set[0] = 0 # 0 represents "Empty" / "Mask"
            
            # Input: The sequence with the hole
            # Target: The sequence filled in (or just the missing val)
            inp = torch.tensor(full_set).long()
            
            # We want to predict the missing value at the position of the 0
            tgt = torch.full_like(inp, -100) # Ignore everything
            tgt[0] = missing_val # Only predict the missing spot
            
            self.data.append((inp, tgt))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- TASK 2: MULTIPLICATION ---
# Input: "12 * 3 ="
# Target: "36"
class MultiplicationDataset(Dataset):
    def __init__(self, samples=5000, max_digits=2):
        self.data = []
        
        # Calculate max sequence length to ensure consistent batching
        # Eq: "99 * 99 = 9801" -> 2 + 1 + 2 + 1 + 4 = 10 tokens
        # We add a buffer to be safe.
        self.max_len = (max_digits * 4) + 4 
        
        # Tokens: 0-9 (Digits), 10 (*), 11 (=), 12 (Pad)
        self.PAD_TOKEN = 12
        
        for _ in range(samples):
            a = random.randint(0, 10**max_digits - 1)
            b = random.randint(0, 10**max_digits - 1)
            c = a * b
            
            # Create sequence: [Digits A, 10, Digits B, 11, Digits C]
            # We convert numbers to list of digits
            seq_a = [int(d) for d in str(a)]
            seq_b = [int(d) for d in str(b)]
            seq_c = [int(d) for d in str(c)]
            
            # Input: "A * B ="
            input_seq = seq_a + [10] + seq_b + [11]
            
            # Target: Mask input, predict C
            target_seq = [-100] * len(input_seq) + seq_c
            
            # Combine Input
            # We need to append the answer to the input for the model to "write" over
            # But for a standard causal model/TRM, we feed inputs and expect next token
            # For this simple test, we feed "A * B =" and expect it to output "C" at the end.
            # To make it compatible with the fixed-size TRM, we pad the input to max_len.
            
            # Pad Input to max_len
            pad_len = self.max_len - len(input_seq)
            if pad_len < 0: continue # Skip if too long (rare)
            
            input_tensor = torch.tensor(input_seq + [0] * len(seq_c) + [self.PAD_TOKEN] * pad_len).long()
            
            # Pad Target
            # We want to predict C. The previous tokens are masked.
            # We align C to be predicted after the '=' sign.
            target_tensor = torch.tensor(target_seq + [-100] * pad_len).long()
            
            self.data.append((input_tensor, target_tensor))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- RUNNER ---
def run_task(name, dataset_cls, epochs=15, num_tokens=17, lr=1e-3, **kwargs):
    print(f"\n[TEST] Starting Task: {name}")
    
    # Handle kwargs specifically to avoid collisions
    train_samples = kwargs.pop('samples', 5000)
    
    # 1. Create Datasets
    train_dataset = dataset_cls(samples=train_samples, **kwargs)
    val_dataset = dataset_cls(samples=100, **kwargs)

    sample_inp, _ = train_dataset[0]
    actual_seq_len = sample_inp.shape[0]
    print(f"      Sequence Length: {actual_seq_len} (Fixed via Padding)")

    # 2. Model Setup
    model = TinyRecursiveModel(
        dim = 128,
        num_tokens = num_tokens,
        max_seq_len = actual_seq_len + 16,
        network = MLPMixer1D(
            dim = 128, 
            depth = 2, 
            seq_len = actual_seq_len 
        ),
        num_refinement_blocks = 2, 
        num_latent_refinements = 4 
    )

    # 3. Trainer
    trainer = Trainer(
        model,
        train_dataset,
        learning_rate = lr,
        weight_decay = 0.0,   # CRITICAL for Logic/Memory
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 12,
        warmup_steps = 10,
        compile_model = False, # Disable compile to prevent inference crashes
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()
    
    # 4. Validation
    print(f"      Validating {name}...")
    inp, tgt = val_dataset[0]
    inp = inp.unsqueeze(0).to(trainer.accelerator.device)
    
    pred, _ = model.predict(inp)
    
    # Check simple accuracy (exact match of valid tokens)
    # Mask out -100 from target comparison
    pred= pred[0]
    tgt_gpu = tgt.to(trainer.accelerator.device)
    mask = tgt_gpu != -100
    
    pred_relevant = pred[mask]
    tgt_relevant = tgt_gpu[mask]
    
    if len(tgt_relevant) == 0:
        print("      >>> WARNING: No valid targets found in sample.")
    else:
        print(f"      Prediction Sample: {pred_relevant.tolist()}")
        print(f"      Target Sample:     {tgt_relevant.tolist()}")
        
        is_correct = (pred_relevant == tgt_relevant).all()
        
        if is_correct:
            print(f"      >>> SUCCESS: {name} Passed.")
        else:
            print(f"      >>> FAILURE: {name} Failed.")

if __name__ == "__main__":
    # Task 1: Set Completion (Sudoku-Lite)
    # Tokens: 0-16 (17 total). LR 1e-3. 
    run_task("Set Completion (Sudoku-Lite)", SetCompletionDataset, epochs=20, num_tokens=17, lr=1e-3, samples=5000, set_size=16)
    
    # Task 2: Multiplication (Hard)
    # Tokens: 0-9 (digits), 10(*), 11(=), 12(pad) -> 13 total.
    run_task("Multiplication (Hard)", MultiplicationDataset, epochs=30, num_tokens=13, lr=1e-3, samples=10000, max_digits=2)