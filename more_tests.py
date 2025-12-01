import torch
from torch.utils.data import Dataset, DataLoader
from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.trainer import Trainer

print("--- ADVANCED REASONING SUITE ---")

class AdditionDataset(Dataset):
    def __init__(self, samples=5000, seq_len=8):
        self.seq_len = seq_len
        self.samples = samples
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        len1 = torch.randint(1, self.seq_len // 2, (1,)).item()
        len2 = torch.randint(1, self.seq_len // 2, (1,)).item()
        n1 = torch.randint(0, 10, (len1,))
        n2 = torch.randint(0, 10, (len2,))
        val1 = int("".join(map(str, n1.tolist())))
        val2 = int("".join(map(str, n2.tolist())))
        res_str = str(val1 + val2)
        res = torch.tensor([int(c) for c in res_str])
        plus, eq = torch.tensor([10]), torch.tensor([11])
        inp = torch.cat((n1, plus, n2, eq))
        pad_len = (self.seq_len * 2) - len(inp) - len(res)
        if pad_len < 0: return self.__getitem__(idx)
        inp = torch.cat((inp, torch.full((pad_len,), 12)))
        tgt = torch.full_like(inp, -100)
        full_inp = torch.cat((inp, res))
        full_tgt = torch.cat((tgt, res))
        return full_inp[:-1], full_tgt[1:]

class SortingDataset(Dataset):
    def __init__(self, samples=5000, seq_len=10):
        self.seq_len = seq_len
        self.samples = samples
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        data = torch.randint(0, 10, (self.seq_len,))
        sorted_data, _ = torch.sort(data)
        delim = torch.tensor([10])
        inp = torch.cat((data, delim, sorted_data))
        tgt = torch.full((len(data) + 1,), -100)
        full_tgt = torch.cat((tgt, sorted_data))
        return inp[:-1], full_tgt[1:]

def run_task(name, dataset, epochs=15, num_tokens=13, lr=1e-3):
    print(f"\n[TEST] Starting Task: {name}")
    sample_inp, _ = dataset[0]
    actual_seq_len = sample_inp.shape[0]
    
    model = TinyRecursiveModel(
        dim = 128,
        num_tokens = num_tokens,
        max_seq_len = actual_seq_len + 32,
        network = MLPMixer1D(dim = 128, depth = 2, seq_len = actual_seq_len),
        num_refinement_blocks = 1, 
        num_latent_refinements = 1 
    )

    trainer = Trainer(
        model,
        dataset,
        learning_rate = lr,
        weight_decay = 0.0,
        batch_size = 32,
        epochs = epochs,
        max_recurrent_steps = 12,
        warmup_steps = 10,
        compile_model = True,
        cpu = not torch.cuda.is_available()
    )

    trainer.forward()
    
    print(f"      Validating {name}...")
    val_loader = DataLoader(dataset, batch_size=32, shuffle=True)
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
    print(f"      Batch Exact Match Accuracy: {accuracy:.2f}%")
    if accuracy > 90.0: print(f"      >>> SUCCESS: {name} Passed.")
    else: print(f"      >>> FAILURE: {name} Failed.")

if __name__ == "__main__":
    run_task("Integer Addition", AdditionDataset(samples=5000, seq_len=8), epochs=20, num_tokens=13, lr=1e-3)
    run_task("List Sorting", SortingDataset(samples=5000, seq_len=10), epochs=20, num_tokens=11, lr=1e-3)