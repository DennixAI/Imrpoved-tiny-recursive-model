import os
import json
import requests
import torch
from torch.utils.data import Dataset
from pathlib import Path

# 1. Download the Data
DATA_DIR = Path("ARC-AGI/data/training")
DATA_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/"

def download_arc():
    if DATA_DIR.exists():
        print("ARC Data already exists.")
        return

    print("Downloading Official ARC-AGI Dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # We need to fetch the file list first (simulated here for standard files)
    # For a real script, usually people clone the repo. 
    # Here is a simple way to get a few tasks to verify.
    
    # Let's verify by checking 5 random known files (ARC has 400 json files)
    # In a real setup: !git clone https://github.com/fchollet/ARC-AGI.git
    print("Please run: 'git clone https://github.com/fchollet/ARC-AGI.git' in your terminal.")
    print("This requires the full repo structure.")

# 2. The Loader
class ARCDataset(Dataset):
    def __init__(self, data_path="ARC-AGI/data/training", augment=True):
        self.files = list(Path(data_path).glob("*.json"))
        self.augment = augment
        print(f"Found {len(self.files)} ARC Tasks.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ARC Data Structure:
        # { "train": [ {"input": [[0,1],...], "output": [[0,1],...]} ], "test": ... }
        with open(self.files[idx], 'r') as f:
            task = json.load(f)
        
        # Getting a Training Pair from the task
        pair = task['train'][0] # Just grab the first example for now
        
        inp_grid = torch.tensor(pair['input']).long()
        out_grid = torch.tensor(pair['output']).long()
        
        # Flatten for the TRM (Sequence Modeling)
        # Note: In reality, you need to pad these to the same size (e.g. 30x30)
        # because ARC grids vary in size.
        return inp_grid.flatten(), out_grid.flatten()

# Usage:
# Run `git clone https://github.com/fchollet/ARC-AGI.git` first!
# dataset = ARCDataset()
