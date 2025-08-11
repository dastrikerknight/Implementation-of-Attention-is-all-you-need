
import torch

class Translator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def translate(self, sentence, max_len=100):
        sp_src, sp_trg = self.tokenizer.spm_src, self.tokenizer.spm_trg

        src_ids = [sp_src.bos_id()] + sp_src.encode(sentence) + [sp_src.eos_id()]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        src_mask = (src_tensor != sp_src.pad_id()).unsqueeze(1).unsqueeze(2)

        trg_ids = [sp_trg.bos_id()]
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_ids], dtype=torch.long).to(self.device)
            trg_pad_mask = (trg_tensor != sp_trg.pad_id()).unsqueeze(1).unsqueeze(2)
            trg_len = trg_tensor.size(1)
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool, device=self.device))
            trg_mask = trg_pad_mask & trg_sub_mask.unsqueeze(0).unsqueeze(1)

            with torch.no_grad():
                output = self.model(src_tensor, trg_tensor, src_mask, trg_mask)
                next_token = output.argmax(-1)[:, -1].item()

            trg_ids.append(next_token)
            if next_token == sp_trg.eos_id():
                break

        return sp_trg.decode(trg_ids[1:-1])
