import os
import copy
import pickle
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from utils import data_cuda


class Trainer:

    def __init__(self, *, args=None, logger=None, logfolder=None, 
        model=None, loss=None, optimizer=None, 
        train_loader=None, valid_loader=None, train_cont_idx=0):

        self.args = args
        self.logger = logger
        self.logfolder = logfolder
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        #self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_cont_idx = train_cont_idx

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        valid_ppl = []
        best_ppl = 1e9
        num_iter = 0

        for epoch in trange(self.args.epochs, desc='Epoch'):
            epoch_loss = 0.0
            epoch_tokens = 0.0
            #iter_ppl = 0.0
            # compute start idx 
            if epoch == 0: 
                # 1st epoch, may start from pre-saved idx 
                start_idx = self.train_cont_idx
            else: 
                # other epoches, must start from 0 
                start_idx = 0

            for i in tqdm(range(start_idx, len(self.train_loader) - 1, self.args.bptt), desc='Iteration'):
                # move to GPU 
                data, target, attn_mask, target_mask = data_cuda(
                    self.train_loader.get_batch(i))
                outputs = self.model(data, attn_mask)
                loss = self.loss(outputs, target)

                # track loss 
                n_tokens = float(target_mask.sum())
                epoch_loss += float(loss.detach()) * n_tokens
                epoch_tokens += n_tokens

                # iteration + 1
                num_iter += 1

                # backprop
                loss.backward()
                
                if num_iter % self.args.update_interval == 0: 
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    # do we need grad clipping
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if num_iter % self.args.log_interval == 0: 
                    # after every log_interval, validate 
                    self.logger.info(f"[Validate] iter={num_iter} & epoch={epoch} & i={i}")
                    valid_ppl_checkpoint = self.eval()
                    self.model.train()
                    valid_ppl += [valid_ppl_checkpoint]

                    model_state = self.model.state_dict()
                    optim_state = self.optimizer.state_dict()
                    "NOTE: this only works for Adam optimizer, NoamOpt not implemented"

                    torch.save(model_state, 
                        os.path.join(self.logfolder, 'model_last'))
                    torch.save(optim_state, 
                        os.path.join(self.logfolder, 'optim_last'))
                    torch.save({'train_cont_idx': i}, 
                        os.path.join(self.logfolder, 'data_last'))

                    if valid_ppl_checkpoint < best_ppl:
                        best_ppl = valid_ppl_checkpoint
                        torch.save(model_state, os.path.join(self.logfolder, 'model_best'))

            self.logger.info(f'Epoch: {epoch:2} | loss: {epoch_loss / epoch_tokens:2.6f}')
            self.logger.info('\n')


    def eval(self): 
        self.model.eval()
        ppl = []
        epoch_loss = 0.0 
        epoch_tokens = 0.0
        with torch.no_grad(): 
            for i in tqdm(range(0, len(self.valid_loader) - 1, self.args.bptt), desc='Eval'): 
                # move to GPU 
                data, target, attn_mask, target_mask = data_cuda(
                    self.valid_loader.get_batch(i))
                outputs = self.model(data, attn_mask)
                loss = self.loss(outputs, target)
                """
                compute perplexity
                """
                n_tokens = float(target_mask.sum())
                epoch_loss += float(loss) * n_tokens
                epoch_tokens += n_tokens

                #del sent, attn_mask, target_mask, outputs, loss
                #torch.cuda.empty_cache()
        ppl = np.exp(epoch_loss / epoch_tokens)
        self.logger.info(f"[EVAL] Valid set | perplexity: {ppl}")
        self.logger.info(f"\n")
        return ppl

