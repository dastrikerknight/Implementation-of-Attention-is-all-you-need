import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

class Data:
    def __init__(self, tokenizer, batch=32, max_len=100):
        self.tokenizer = tokenizer
        self.batch = batch
        self.max_len = max_len

    def collate(self, batch):
        spm_src, spm_trg = self.tokenizer.spm_src, self.tokenizer.spm_trg
        srcbatch, trgbatch = [], []

        for text in batch:
            src_ids = [spm_src.bos_id()] + spm_src.encode(text["en"]) + [spm_src.eos_id()]
            trg_ids = [spm_trg.bos_id()] + spm_trg.encode(text["de"]) + [spm_trg.eos_id()]
            srcbatch.append(src_ids[:self.max_len])
            trgbatch.append(trg_ids[:self.max_len])

        src_max = max(len(s) for s in srcbatch)
        trg_max = max(len(t) for t in trgbatch)

        src_padded = [s + [spm_src.pad_id()] * (src_max - len(s)) for s in srcbatch]
        trg_padded = [t + [spm_trg.pad_id()] * (trg_max - len(t)) for t in trgbatch]

        src_tensor = torch.tensor(src_padded, dtype=torch.long)
        trg_tensor = torch.tensor(trg_padded, dtype=torch.long)

        src_mask = (src_tensor != spm_src.pad_id()).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg_tensor != spm_trg.pad_id()).unsqueeze(1).unsqueeze(2)
        trg_len = trg_tensor.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool))
        trg_mask = trg_pad_mask & trg_sub_mask.unsqueeze(0).unsqueeze(1)

        return src_tensor, trg_tensor, src_mask, trg_mask

    def dataloader(self):
        dataset = load_dataset("multi30k", split={"train": "train", "validation": "validation", "test": "test"})
        train_loader = DataLoader(dataset["train"], batch_size=self.batch, shuffle=True, collate_fn=self.collate)
        val_loader = DataLoader(dataset["validation"], batch_size=self.batch, shuffle=False, collate_fn=self.collate)
        test_loader = DataLoader(dataset["test"], batch_size=self.batch, shuffle=False, collate_fn=self.collate)
        return train_loader, val_loader, test_loader
