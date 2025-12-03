import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer
import matplotlib.pyplot as plt
try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

class Sudoku1MDataset(Dataset):
    def __init__(self, split="train", limit=10000):
        self.data = []
        print(f"Loading {limit} puzzles...")
        loaded = False
        
        # Try HuggingFace
        if HAS_HF and not loaded:
            try:
                print("Attempting to load 'Ritvik19/Sudoku-Dataset'...")
                ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train", streaming=True)
                count = 0
                for item in ds:
                    if count >= limit: break
                    quiz = np.array([int(c) for c in item['puzzle']])
                    solution = np.array([int(c) for c in item['solution']])
                    self.data.append((torch.tensor(quiz).long(), torch.tensor(solution).long()))
                    count += 1
                loaded = True
            except Exception as e:
                print(f"HF Load failed: {e}")

        # Try CSV
        if not loaded:
            print("HF failed. Checking for local 'sudoku.csv'...")
            if os.path.exists("sudoku.csv"):
                try:
                    df = pd.read_csv("sudoku.csv", nrows=limit)
                    for _, row in df.iterrows():
                        q_col = 'quizzes' if 'quizzes' in df.columns else 'puzzle'
                        s_col = 'solutions' if 'solutions' in df.columns else 'solution'
                        quiz = np.array([int(c) for c in str(row[q_col])])
                        solution = np.array([int(c) for c in str(row[s_col])])
                        self.data.append((torch.tensor(quiz).long(), torch.tensor(solution).long()))
                    loaded = True
                except Exception as e:
                    print(f"CSV Read failed: {e}")
            else:
                print("Error: No data found. Please download sudoku.csv.")
                exit()

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def run_sudoku_benchmark():
    print("--- BENCHMARK: 9x9 SUDOKU ---")
    
    # Use 50k for a decent overnight test, or 1k for a quick plot check
    train_ds = Sudoku1MDataset(split="train", limit=50000)
    val_ds = Sudoku1MDataset(split="train", limit=100) 
    model = TinyRecursiveModel(
        dim = 384,              
        num_tokens = 10,        
        max_seq_len = 81 + 16,  
        network = MLPMixer1D(
            dim = 384, 
            depth = 6,          
            seq_len = 81 
        ),
        num_refinement_blocks = 4, 
        num_latent_refinements = 4 
    )

    trainer = Trainer(
        model,
        train_ds,
        learning_rate = 1e-3,
        weight_decay = 0.0,
        batch_size = 8,
        accelerate_kwargs = {
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 4
        },
        epochs = 10,
        max_recurrent_steps = 32,
        warmup_steps = 1000,
        compile_model = False,
        cpu = not torch.cuda.is_available()
    )

    # get loss history
    print("Starting Training...")
    history = trainer.forward()
    
    # plotting
    print("Saving Loss Plot to 'sudoku_loss.png'...")
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("Recursive Sudoku Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("sudoku_loss.png")
    print("Plot saved.")

    print("Validating Sudoku...")
    correct_puzzles = 0
    total = len(val_ds)
    
    for i in range(total):
        inp, tgt = val_ds[i]
        inp_gpu = inp.unsqueeze(0).to(trainer.accelerator.device)
        pred, _ = model.predict(inp_gpu)
        pred = pred[0]
        
        if (pred == tgt.to(trainer.accelerator.device)).all():
            correct_puzzles += 1
            
    print(f"Result: {correct_puzzles}/{total} Puzzles Solved Perfectly")

if __name__ == "__main__":
    run_sudoku_benchmark()