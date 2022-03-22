import os
import copy
import pickle
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from utils import data_cuda
import gc


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
                # track and clear memory
                gc.collect()
                torch.cuda.empty_cache()
                #torch.cuda.reset_max_memory_allocated()

                # move to GPU 
                data, target, attn_mask, target_mask = data_cuda(
                    self.train_loader.get_batch(i))
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
                    loss = self.loss(outputs, target)

                # track loss 
                n_tokens = float(target_mask.sum())
                epoch_loss += float(loss.detach()) * n_tokens
                epoch_tokens += n_tokens

                # iteration + 1
                num_iter += 1

                # backprop
                loss.backward()

                # clip gradient 
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_norm)
                
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
                        """
                        what if we want to early stop on bpc but not ppl? 
                        it is actually the same 
                        since ppl = exp(log(2) x bpc) --- monotonic relation!
                        therefore, we do not need to bother writing a cri-specific checkpoint
                        """
                        best_ppl = valid_ppl_checkpoint
                        torch.save(model_state, os.path.join(self.logfolder, 'model_best'))

                # track memory
                #torch.cuda.synchronize()
                #print("\nMax memory used by tensors")
                #xxx = torch.cuda.max_memory_allocated()
                #ggg = xxx//1000000000
                #mmm = (xxx-ggg*1000000000) // 1000000
                #print(f"{xxx}B")
                #print(f"{ggg}GB-{mmm} MB")
                #exit(0)

            self.logger.info(f'Epoch: {epoch:2} | loss: {epoch_loss / epoch_tokens:2.6f}')
            self.logger.info('\n')


    def eval(self): 
        self.model.eval()
        ppl = []
        epoch_loss = 0.0 
        epoch_tokens = 0.0
        with torch.no_grad(): 
            for i in tqdm(range(0, len(self.valid_loader) - 1, self.args.bptt), desc='Eval'): 
                # track and clear memory
                gc.collect()
                torch.cuda.empty_cache()
                #torch.cuda.reset_max_memory_allocated()

                # move to GPU 
                data, target, attn_mask, target_mask = data_cuda(
                    self.valid_loader.get_batch(i))
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
                    loss = self.loss(outputs, target)
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
        self.logger.info(f"[EVAL] Valid set | perplexity: {ppl} | bit per char: {bpc}")
        self.logger.info(f"\n")
        return ppl
