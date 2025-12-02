import torch
import random
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- EXPERT SUITE: SET COMPLETION & ARITHMETIC ---")

# --- TASK 1: SET COMPLETION (Sudoku-Lite) ---
class SetCompletionDataset(Dataset):
    def __init__(self, samples=5000, set_size=16):
        self.data = []
        self.set_size = set_size
        
        for _ in range(samples):
            full_set = list(range(1, set_size + 1))
            random.shuffle(full_set)
            
            missing_val = full_set[0]
            full_set[0] = 0 
            
            inp = torch.tensor(full_set).long()
            tgt = torch.full_like(inp, -100) 
            tgt[0] = missing_val 
            
            self.data.append((inp, tgt))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- TASK 2: MULTIPLICATION (FIXED PADDING) ---
class MultiplicationDataset(Dataset):
    def __init__(self, samples=5000, max_digits=2):
        self.data = []
        
        # Fixed Layout: [A_digits] [*] [B_digits] [=] [C_digits] [Padding]
        # Max A=2, B=2, C=4. Symbols=2. Total ~10. 
        # Let's set a hard limit that fits everything.
        self.FIXED_LEN = (max_digits * 4) + 4 # approx 12-16
        self.PAD_TOKEN = 12
        
        for _ in range(samples):
            a = random.randint(0, 10**max_digits - 1)
            b = random.randint(0, 10**max_digits - 1)
            c = a * b
            
            seq_a = [int(d) for d in str(a)]
            seq_b = [int(d) for d in str(b)]
            seq_c = [int(d) for d in str(c)]
            
            # 1. Build Content List
            # Input: 1 2 * 3 = 0 0 (0s are placeholders for answer)
            input_content = seq_a + [10] + seq_b + [11] + [0] * len(seq_c)
            
            # Target: I I I I I 3 6 (I is ignore)
            target_content = [-100] * (len(seq_a) + 1 + len(seq_b) + 1) + seq_c
            
            # 2. Check fit
            if len(input_content) > self.FIXED_LEN:
                continue 
                
            # 3. Calc Padding
            pad_len = self.FIXED_LEN - len(input_content)
            
            # 4. Construct Tensors
            inp = torch.tensor(input_content + [self.PAD_TOKEN] * pad_len).long()
            tgt = torch.tensor(target_content + [-100] * pad_len).long()
            
            self.data.append((inp, tgt))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def run_task(name, dataset_cls, epochs=15, num_tokens=17, lr=1e-3, **kwargs):
    print(f"\n[TEST] Starting Task: {name}")
    
    train_samples = kwargs.pop('samples', 5000)
    
    train_dataset = dataset_cls(samples=train_samples, **kwargs)
    val_dataset = dataset_cls(samples=100, **kwargs)

    sample_inp, _ = train_dataset[0]
    actual_seq_len = sample_inp.shape[0]
    print(f"      Sequence Length: {actual_seq_len} (Fixed)")

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

    trainer = Trainer(
        model,
        train_dataset,
        learning_rate = lr,
        weight_decay = 0.0,
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 12,
        warmup_steps = 10,
        compile_model = False, 
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()

    print(f"      Validating {name}...")
    inp, tgt = val_dataset[0]
    inp = inp.unsqueeze(0).to(trainer.accelerator.device)
    
    pred, _ = model.predict(inp)
    pred = pred[0] 
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
    run_task("Set Completion (Sudoku-Lite)", SetCompletionDataset, epochs=20, num_tokens=17, lr=1e-3, samples=5000, set_size=16)
    
    # Task 2: Multiplication (Hard)
    # Using LR=5e-4 to be safe with deeper logic
    run_task("Multiplication (Hard)", MultiplicationDataset, epochs=30, num_tokens=13, lr=5e-4, samples=10000, max_digits=2)