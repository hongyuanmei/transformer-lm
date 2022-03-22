import os
import copy
import pickle
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from utils import data_cuda
import gc


class Tester:

    def __init__(self, *, args=None, model=None, criterion=None, data_loader=None):

        self.args = args
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader

    def eval(self): 
        self.model.eval()
        ppl = []
        epoch_loss = 0.0 
        epoch_tokens = 0.0
        with torch.no_grad(): 
            for i in tqdm(range(0, len(self.data_loader) - 1, self.args.bptt), desc='Eval'): 
                # track and clear memory
                gc.collect()
                torch.cuda.empty_cache()
                #torch.cuda.reset_max_memory_allocated()

                # move to GPU 
                data, target, attn_mask, target_mask = data_cuda(
                    self.data_loader.get_batch(i))
                outputs = self.model(data, attn_mask)

                if self.args.ada: 
                    # generator is adaptive softmax
                    od1, od2, od3 = outputs.size()
                    outputs, loss = self.model.generator(
                        outputs.contiguous().view(-1, od3), 
                        target.contiguous().view(-1)
                    )
                else: 
                    # generator is linear
                    outputs = self.model.generator(outputs)
                    loss = self.criterion(outputs, target)
                """
                compute perplexity
                """
                n_tokens = float(target_mask.sum())
                epoch_loss += float(loss) * n_tokens
                epoch_tokens += n_tokens

                #del sent, attn_mask, target_mask, outputs, loss
                #torch.cuda.empty_cache()
        
        avgneglogprob = epoch_loss / epoch_tokens
        ppl = np.exp(avgneglogprob)
        bpc = avgneglogprob / np.log(2.0)
        # ppl = exp(log(2) x bpc)
        print(f"[EVAL ({self.args.split})] Valid set | perplexity: {ppl} | bit per char: {bpc}")
        print(f"\n")
        return ppl
