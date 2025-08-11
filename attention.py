# attention.py
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=-1) 
        _2i = torch.arange(0, d_model, step=2, device=device).float()  

        div_term = 10000 ** (_2i / d_model)
        self.encoding[:, 0::2] = torch.sin(pos / div_term)
        self.encoding[:, 1::2] = torch.cos(pos / div_term)

    def forward(self, x):
        if x.dim() == 2:
            batch, seq_len = x.size()
        else:
            batch, seq_len, _ = x.size()
        pe = self.encoding[:seq_len, :]            
        pe = pe.unsqueeze(0).expand(batch, -1, -1)   
        return pe

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, eps=1e-9):
        
        batch, head, q_len, d_k = q.size()
       
        kt = k.transpose(2, 3)
      
        score = (q @ kt) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e9'))

        attn = self.softmax(score)  
        output = attn @ v          
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        
        batch, q_len, _ = q.size()
        _, k_len, _ = k.size()
        
        q = self.q_linear(q)  
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = self._split_heads(q)  
        k = self._split_heads(k)  
        v = self._split_heads(v)  

       
        out, attn = self.attention(q, k, v, mask=mask) 

     
        out = self._concat_heads(out)  
        out = self.out_linear(out)
        return out

    def _split_heads(self, x):
       
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_head, self.d_k)
        x = x.transpose(1, 2)                             
        return x

    def _concat_heads(self, x):
        
        batch, head, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, seq_len, head * d_k)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden, dropprob=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropprob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(max_len, d_model, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        
        token_emb = self.token_embed(x)           
        pos_emb = self.pos_embed(x)                
        return self.dropout(token_emb + pos_emb)

class EncodeLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncodeLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = FeedForwardNetwork(d_model=d_model, hidden=ffn_hidden, dropprob=drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.self_attn(x, x, x, mask=src_mask)  
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, enc_vocabsize):
        super(Encoder, self).__init__()
        self.embed = TransformerEmbedding(vocab_size=enc_vocabsize, d_model=d_model, max_len=max_len, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([EncodeLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])

    def forward(self, src, src_mask):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecodeLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecodeLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = FeedForwardNetwork(d_model=d_model, hidden=ffn_hidden, dropprob=drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, enc_output, trg_mask, src_mask):
        _x=x
        x = self.self_attn(x, x, x, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, max_len, drop_prob, device, dec_vocabsize, ffn_hidden):
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(vocab_size=dec_vocabsize, d_model=d_model, max_len=max_len, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([DecodeLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])
        self.out_linear = nn.Linear(d_model, dec_vocabsize)

    def forward(self, trg, enc_output, src_mask, trg_mask):
        x = self.embed(trg)
        for layer in self.layers:
            x = layer(x, enc_output, trg_mask, src_mask)
        output = self.out_linear(x)
        return output

class Transformer(nn.Module):
    def __init__(self, src_pad_indx, trg_pad_indx, trg_sos_indx,
                 enc_vocabsize, dec_vocabsize, d_model, n_head,
                 max_len, ffn, n_layers, dropprobab, device):
        super().__init__()
        self.src_pad_indx = src_pad_indx
        self.trg_pad_indx = trg_pad_indx
        self.trg_sos_indx = trg_sos_indx
        self.device = device

       
        self.encoder = Encoder(max_len=max_len, d_model=d_model, ffn_hidden=ffn, n_head=n_head, n_layers=n_layers, drop_prob=dropprobab, device=device, enc_vocabsize=enc_vocabsize)
        self.decoder = Decoder(d_model=d_model, n_head=n_head, n_layers=n_layers, max_len=max_len, drop_prob=dropprobab, device=device, dec_vocabsize=dec_vocabsize, ffn_hidden=ffn)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)
        return output
