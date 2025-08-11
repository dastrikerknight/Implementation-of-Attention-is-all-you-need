import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tokenizer import Tokenizer
from dataloader import Data
from attention import Transformer
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dummy_data = [
    {"en": "Hello world", "de": "Hallo Welt"},
    {"en": "How are you", "de": "Wie geht es dir"},
    {"en": "I love cats", "de": "Ich liebe Katzen"},
    {"en": "A man rides a horse", "de": "Ein Mann reitet ein Pferd"},
] * 10  

src_texts = [ex["en"] for ex in dummy_data]
trg_texts = [ex["de"] for ex in dummy_data]


tok = Tokenizer()
tok.train(src_texts, trg_texts)
SRC_PAD_IDX, TRG_PAD_IDX = tok.padindx
TRG_SOS_IDX, _ = tok.sos_eos_indx


data = Data(tok, batch=4, max_len=20)

dummy_loader = DataLoader(dummy_data, batch_size=4, shuffle=True, collate_fn=data.collate)

model = Transformer(
    src_pad_indx=SRC_PAD_IDX,
    trg_pad_indx=TRG_PAD_IDX,
    trg_sos_indx=TRG_SOS_IDX,
    enc_vocabsize=tok.spm_src.get_piece_size(),
    dec_vocabsize=tok.spm_trg.get_piece_size(),
    d_model=64, n_head=4, max_len=20, ffn=128,
    n_layers=2, dropprobab=0.1, device=DEVICE
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
trainer = Trainer(model, optimizer, criterion, DEVICE)

print("=== Running dummy training loop ===")
train_loss = trainer.train_epoch(dummy_loader)
val_loss = trainer.evaluate(dummy_loader)
print(f"Dummy Train Loss: {train_loss:.3f}, Dummy Val Loss: {val_loss:.3f}")
