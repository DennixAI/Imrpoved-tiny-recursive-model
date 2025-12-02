import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

# This is the dataset used by many researchers (1 million puzzles)
# It contains 'quizzes' (input) and 'solutions' (target)
print("Downloading Sudoku 1M Dataset from HuggingFace...")
dataset_stream = load_dataset("engineering-org/sudoku_1m", split="train", streaming=True)

class RealSudokuDataset(Dataset):
    def __init__(self, samples=5000):
        self.data = []
        print(f"Loading {samples} puzzles into memory...")
        
        # Take the first N samples from the stream
        iterator = iter(dataset_stream)
        for _ in range(samples):
            item = next(iterator)
            
            # Convert string "004300..." to tensor [0, 0, 4, 3, 0...]
            # 'quizzes' is the unsolved puzzle (Input)
            # 'solutions' is the solved puzzle (Target)
            quiz = np.array([int(c) for c in item['quiz']])
            sol = np.array([int(c) for c in item['solution']])
            
            # +1 because our model uses 0 for padding/masking usually, 
            # but Sudoku uses 0 for empty. 
            # Let's map 0->0 (Empty) and 1-9 -> 1-9.
            # (If your model needs non-zero tokens, map 0->10)
            self.data.append((
                torch.tensor(quiz, dtype=torch.long),
                torch.tensor(sol, dtype=torch.long)
            ))
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# Usage in your main script:
# dataset = RealSudokuDataset(samples=10000)
