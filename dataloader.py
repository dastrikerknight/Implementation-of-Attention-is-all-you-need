import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm
from * import 


class Data:
  def __init__(self,tokenizer,batch,max_len):
    self.tokenizer= tokenizer
    self.batch= batchsize
    self.max_len= max_len

def collate(self,batch):
  self.spm_src= self.tokenizer.spm_src
  self.spm_trg= self.tokenizer.spm_trg
  srcbatch=[]
  trgbatch=[]

for text in batch:
  srcindx=[spm_src.bos_id()]+spm_src.encode(text["en"]) + [spm_src.eos_id()]
  trgindx=[spm_trg.bos_id()]+spm_trg.encode(text["de"])+ [spm_trg.eos_id()]
  srcbatch.append(srcindx[:self.max_len])
  trgbatch.append(trgindx[:self.max_len])

srcmax= max(len(a) for a in srcbatch)
trgmax= max(len(b) for b in trgbatch)
srcpadded=[a+[spm_src.pad_id()]* (srcmax-len(a)) for a in batch]
trgpadded= [b+[spm_trg.pad_id()] * (trgmax-len(b)) for b in batch]

srctensor= torch.tensor(srcpadded, dtype= torch.long)
trgtensor= torch.tensor(trgpadded, dtype= torch.long)

src_mask= (srctensor!= spm_src.pad_id()).unsqueeze(1).unsqueeze(2)
trg_pad_mask= (trgtensor!= spm_trg.pad_id()).unsqueeze(1).unsqueeze(2)

trg_len= trgtensor.size(1)
trg_sub_mask= torch.tril(torch.ones(trg_len, trg_len),dtype=torch.bool))
trg_mask = trg_pad_mask & trg_sub_mask.unsueeze(0).unsqueeze(1)
return src_tensor,trg_tensor,src_mask,trg_mask

def dataloader(self):
  print("Loading dataset....")
  dataset= load_dataset("multi30k",split={"train":"train","validation":"validation","test":"test"})
  train_loader= DataLoader(dataset['train'],batch_size=self.batch,shuffle=True,collate_fn=self.collate)
  test_loader= DataLoader(dataset['test'],batch_size=self.batch,shuffle=False,collate_fn=self.collate)
  validation_loader= DataLoader(dataset['validation'],batch_size=self.batch,shuffle=False,collate_fn=self.collate)
  return train_loader,test_loader,validation_loader
            
