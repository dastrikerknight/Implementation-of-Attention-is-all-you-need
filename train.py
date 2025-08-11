import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from tokenizer import tokenizer
from dataloader import Data
from attention import Transformer
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = load_dataset("multi30k", split="train")
src_texts = [ex["en"] for ex in dataset_train]
trg_texts = [ex["de"] for ex in dataset_train]

tok = tokenizer()
tok.train(src_texts, trg_texts)

SRC_PAD_IDX, TRG_PAD_IDX = tok.pad_ids
TRG_SOS_IDX, _ = tok.sos_eos_ids

data = Data(tok, batch=32, max_len=100)
train_loader, val_loader, test_loader = data.dataloader()


model = Transformer(
    src_pad_indx=SRC_PAD_IDX,
    trg_pad_indx=TRG_PAD_IDX,
    trg_sos_indx=TRG_SOS_IDX,
    enc_vocabsize=tok.spm_src.get_piece_size(),
    dec_vocabsize=tok.spm_trg.get_piece_size(),
    d_model=256, n_head=8, max_len=100, ffn=512,
    n_layers=3, dropprobab=0.1, device=DEVICE
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
trainer = Trainer(model, optimizer, criterion, DEVICE)


train_loss = trainer.train_epoch(train_loader)
val_loss = trainer.evaluate(val_loader)

print(f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
