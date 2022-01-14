import os
import numpy as np
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

"""
for data processing
"""

def get_rawdata(name, split, root): 
    with open(os.path.join(root, f"data/{name}/{split}"), 'r') as f: 
        lines = f.read().split('\n')
    return lines         

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def data_cuda(data_tuple): 
    rst = []
    for x in data_tuple: 
        rst += [x.cuda()]
    return rst

"""
for Transformer LM
"""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None): 
    "scaled dot product attention"
    d_key = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / max(math.sqrt(d_key), 1.0)
    if mask is not None: 
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None: 
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn