import torch
from torch.utils.data import Dataset, DataLoader
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- EXPERT REASONING SUITE ---")

# --- TASK 1: SET COMPLETION (Sudoku Logic) ---
# Input: A set of unique numbers from 0-N, scrambled, with one replaced by a mask.
# Target: The missing number.
# Example (Tokens 0-4): Input [3, 0, 1, 4, MASK] -> Target: 2
class SetCompletionDataset(Dataset):
    def __init__(self, samples=5000, set_size=16):
        self.set_size = set_size
        self.samples = samples
    
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        # Generate full set 0..N-1
        full_set = torch.randperm(self.set_size)
        
        # Pick a number to hide
        hidden_idx = torch.randint(0, self.set_size, (1,)).item()
        target_val = full_set[hidden_idx]
        
        # Replace with MASK (using set_size as the mask token)
        inp = full_set.clone()
        inp[hidden_idx] = self.set_size 
        
        # Target: Ignore everything except the masked position
        tgt = torch.full_like(inp, -100)
        tgt[hidden_idx] = target_val
        
        return inp, tgt

# --- TASK 2: MULTIPLICATION (High Complexity) ---
# Input: "3 * 4 =" -> Output: "12"
# Handled as digit sequences.
class MultiplicationDataset(Dataset):
    def __init__(self, samples=5000, max_digits=3):
        self.samples = samples
        self.max_digits = max_digits
        
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        # Generate two random numbers
        a = torch.randint(0, 10**self.max_digits, (1,)).item()
        b = torch.randint(0, 10**self.max_digits, (1,)).item()
        
        # Calculate result
        res = a * b
        
        # Tokenize (0-9 digits, 10=*, 11==, 12=PAD)
        a_seq = [int(d) for d in str(a)]
        b_seq = [int(d) for d in str(b)]
        res_seq = [int(d) for d in str(res)]
        
        inp_list = a_seq + [10] + b_seq + [11]
        
        # Pad input
        total_len = (self.max_digits * 2) + 2
        padding = total_len - len(inp_list)
        inp_list = inp_list + [12] * padding
        
        inp = torch.tensor(inp_list)
        tgt_seq = torch.tensor(res_seq)
        
        # Full Input: Input + Result
        full_inp = torch.cat((inp, tgt_seq))
        
        # Target: Mask input, predict result
        mask = torch.full_like(inp, -100)
        full_tgt = torch.cat((mask, tgt_seq))
        
        return full_inp[:-1], full_tgt[1:]

def run_task(name, dataset_cls, epochs=15, num_tokens=100, lr=1e-3, **kwargs):
    print(f"\n[TEST] Starting Task: {name}")
    
    # Instantiate dataset
    dataset = dataset_cls(**kwargs)
    sample_inp, _ = dataset[0]
    actual_seq_len = sample_inp.shape[0]
    
    # Setup Model
    model = TinyRecursiveModel(
        dim = 128,
        num_tokens = num_tokens,
        max_seq_len = actual_seq_len + 32,
        network = MLPMixer1D(
            dim = 128, 
            depth = 2, 
            seq_len = actual_seq_len 
        ),
        num_refinement_blocks = 1, 
        num_latent_refinements = 1 
    )

    trainer = Trainer(
        model,
        dataset,
        learning_rate = lr,
        weight_decay = 0.0, # Zero Decay for Logic
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 12,
        warmup_steps = 10,
        compile_model = True,
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()
    
    print(f"      Validating {name}...")
    
    # Check Accuracy on a fresh batch
    val_dataset = dataset_cls(samples=100, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    inp_batch, tgt_batch = next(iter(val_loader))
    
    inp_batch = inp_batch.to(trainer.accelerator.device)
    tgt_batch = tgt_batch.to(trainer.accelerator.device)
    
    preds, _ = model.predict(inp_batch)
    
    mask = tgt_batch != -100
    correct = 0
    total = 0
    
    for i in range(32):
        p = preds[i][mask[i]]
        t = tgt_batch[i][mask[i]]
        
        if p.shape == t.shape and torch.equal(p, t):
            correct += 1
        total += 1
        
    accuracy = (correct / total) * 100
    print(f"      Batch Accuracy: {accuracy:.2f}%")
    
    if accuracy > 90.0:
        print(f"      >>> SUCCESS: {name} Passed.")
    else:
        print(f"      >>> FAILURE: {name} Failed.")

if __name__ == "__main__":
    # Task 1: Set Completion (Sudoku Logic)
    # Set size 16. Vocab: 0-15 are nums, 16 is MASK. Total 17 tokens.
    run_task("Set Completion (Sudoku-Lite)", SetCompletionDataset, epochs=15, num_tokens=17, lr=1e-3, samples=5000, set_size=16)
    
    # Task 2: Multiplication (2 digits * 2 digits)
    # Vocab: 0-9 digits, 10(*), 11(=), 12(PAD). Total 13 tokens.
    run_task("Multiplication (Hard)", MultiplicationDataset, epochs=30, num_tokens=13, lr=1e-3, samples=10000, max_digits=2)