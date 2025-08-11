import torch
import sentencepiece as spm
import os

class Tokenizer:
    def __init__(self,srcfile='spm_src.model',trgfile='spm_trg.model',vocabsize=29):
        self.srcfile= srcfile
        self.trgfile= trgfile
        self.vocabsize= vocabsize
        self.spm_src= spm.SentencePieceProcessor()
        self.spm_trg= spm.SentencePieceProcessor()

    def train(self,src_text,trg_text):
        if not os.path.exists(self.srcfile):
            with open("src.txt",'w',encoding='utf-8') as f:
                f.write("\n".join(src_text))
            with open("trg.txt","w",encoding="utf-8") as f:
                f.write("\n".join(trg_text))

            spm.SentencePieceTrainer.Train(f"--input=src.txt --model_prefix=spm_src --vocab_size={self.vocabsize}"
                                           " --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3")
            spm.SentencePieceTrainer.Train(f"--input=trg.txt --model_prefix=spm_trg --vocab_size={self.vocabsize}"
                                           " --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3")
        self.spm_src.load("spm_src.model")
        self.spm_trg.load("spm_trg.model")

    @property
    def padindx(self):
        return self.spm_src.pad_id(),self.spm_trg.pad_id()
    
    @property
    def sos_eos_indx(self):
        return self.spm_trg.bos_id(),self.spm_trg.eos_id()
    
