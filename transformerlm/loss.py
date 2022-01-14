import torch
import torch.nn as nn 

class NLL: 
    "simple negative log-likelihood loss"
    def __init__(self, pad_idx): 
        self.pad_idx = pad_idx
        self.criterion = nn.NLLLoss(ignore_index=pad_idx)
    
    def __call__(self, x, y): 
        batch_size = x.size(0)
        vocab_size = x.size(-1)
        loss = self.criterion(
            x.contiguous().view(-1, vocab_size), # B x T x V
            y.contiguous().view(-1) # B x T
        )
        return loss 
