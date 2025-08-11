import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc="Training", leave=False)

        for src, trg, src_mask, trg_mask in progress_bar:
            src, trg = src.to(self.device), trg.to(self.device)
            src_mask, trg_mask = src_mask.to(self.device), trg_mask.to(self.device)

        
            trg_input = trg[:, :-1]
            targets = trg[:, 1:]
            trg_mask = trg_mask[:, :, :-1, :-1]

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(src, trg_input, src_mask, trg_mask)
                output_dim = output.shape[-1]
                loss = self.criterion(output.view(-1, output_dim), targets.reshape(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for src, trg, src_mask, trg_mask in progress_bar:
                src, trg = src.to(self.device), trg.to(self.device)
                src_mask, trg_mask = src_mask.to(self.device), trg_mask.to(self.device)

                trg_input = trg[:, :-1]
                targets = trg[:, 1:]
                trg_mask = trg_mask[:, :, :-1, :-1]

                output = self.model(src, trg_input, src_mask, trg_mask)
                output_dim = output.shape[-1]
                loss = self.criterion(output.view(-1, output_dim), targets.reshape(-1))

                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        return total_loss / len(loader)
