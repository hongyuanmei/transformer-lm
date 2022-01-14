import warnings
warnings.filterwarnings('ignore')

import os
import sys
import csv
import json
import random
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import torch
from loss import NLL 
from trainer import Trainer
from tokenizer import Tokenizer
from dataset import Dataset
from model import TransformerLM
from utils import TqdmLoggingHandler, get_rawdata

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=10, type=int, 
        help='epochs to train')
    parser.add_argument('--train_batch_size', default=4, type=int,
        help='train batch size')
    parser.add_argument('--valid_batch_size', default=32, type=int,
        help='valid batch size')
    parser.add_argument('--bptt', default=512, type=int, 
        help='len of seq in each batch')
    parser.add_argument('--log_interval', default=1000, type=int, 
        help='# of training batches between validation')
    parser.add_argument('--update_interval', default=1, type=int, 
        help='# of training batches between param update')
    parser.add_argument('--dataset', default='wikitext-2', type=str, 
        help='dataset')
    parser.add_argument('--n_layers', default=8, type=int, 
        help='num of layers')
    parser.add_argument('--d_emb', default=512, type=int, 
        help='dimension of input word embedding')
    parser.add_argument('--d_model', default=512, type=int, 
        help='dimension of model hidden state')
    parser.add_argument('--n_heads', default=4, type=int, 
        help='num of heads')
    parser.add_argument('--d_ffn', default=1024, type=int, 
        help='dimension of feedforward network')
    parser.add_argument('--d_softmax', default=512, type=int, 
        help='dimension of output word embedding')
    parser.add_argument('--tie_emb', default=1, type=int, choices=[0,1], 
        help='tie input and output embeddings: 0=no, 1=yes')
    parser.add_argument('--dropout', default=0.0, type=float,
        help='dropout rate')
    # parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'noamopt'], 
    #     help='optimizer')
    parser.add_argument('--lr', default=1e-3, type=float,
        help='learning rate, only used for adam')
    #parser.add_argument('--weight_decay', default=1e-5, type=float,
    #    help='weight decay')
    parser.add_argument('--seed', default=12345, type=int, 
        help='random seed')
    parser.add_argument('--cont', default=0, type=int, choices=[0,1], 
        help='continue with saved model and optim: 0=no, 1=yes')
    parser.add_argument('--root_path', default='../', type=str, 
		help='root path of project')

    args = parser.parse_args()

    assert args.log_interval % args.update_interval == 0, \
        f"log_interval ({args.log_interval}) is not a multiple of update_interval ({args.update_interval})"

    # n_gpu = torch.cuda.device_count()
    #if n_gpu > 1 and args.shot == 'all':
    #    args.train_batch_size = args.train_batch_size * n_gpu
    #    args.eval_batch_size = args.eval_batch_size * n_gpu
        
    # name for Log file and checkpoint
    config = []
    for a in vars(args): 
        if a not in ['root_path', 'cont', 'epochs']: 
            # not include useless info in name
            config += [f'{a}={getattr(args, a)}']
    config = '-'.join(config)

    logfolder = os.path.join(args.root_path, f"log/{args.dataset}/{config}")
    os.makedirs(logfolder, exist_ok=True)
    logfile = os.path.join(logfolder, 'log')

    filemode = 'w'
    if args.cont == 1: 
        filemode = 'a' 

    logging.basicConfig(
        format='%(asctime)s - %(name)s -   %(message)s', 
        filename=logfile, filemode=filemode, level=logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().addHandler(TqdmLoggingHandler())
    logger = logging.getLogger(__name__)
    logger.info(config)

    # tokenize data 
    rawdata_train = get_rawdata(args.dataset, 'train', args.root_path)
    rawdata_valid = get_rawdata(args.dataset, 'valid', args.root_path)
    tokenizer = Tokenizer(data=rawdata_train)

    # Dataset
    data_train = Dataset(
        tokenizer.tokenize_data(rawdata_train), 
        args.train_batch_size, args.bptt, tokenizer.get_pad())
    data_valid = Dataset(
        tokenizer.tokenize_data(rawdata_valid), 
        args.valid_batch_size, args.bptt, tokenizer.get_pad())
    
    # set random numbers 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    model = TransformerLM(
        tokenizer.get_size(), 
        args.n_layers, args.d_emb, args.d_model, args.n_heads, 
        args.d_ffn, args.d_softmax,
        args.tie_emb, args.dropout
    )
    #modelfile = os.path.join(logfolder, f'model')
    if args.cont == 1: 
        # read saved model
        model_state = torch.load(os.path.join(logfolder, 'model_last'))
        model.load_state_dict(model_state)
    # move to GPU
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_m = round(total_params / 1000000.0)
    logger.info(f"# of model parameters : {total_params} or {total_params_in_m}M")
    #if torch.cuda.is_available():
    #    model = model.to('cuda')
    
    # loss 
    loss = NLL(tokenizer.get_pad())
    
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cont == 1: 
        # read saved state of optimizer
        "must move model to CUDA before loading optim state"
        optim_state = torch.load(os.path.join(logfolder, 'optim_last'))
        optimizer.load_state_dict(optim_state)
        # read saved dataloader 
        train_cont_idx = torch.load(
            os.path.join(logfolder, 'data_last'))['train_cont_idx']
    else: 
        train_cont_idx = 0

    # Run
    trainer = Trainer(
        args = args, logger = logger, logfolder = logfolder, 
        model = model, loss = loss, optimizer = optimizer, 
        train_loader = data_train, valid_loader = data_valid, 
        train_cont_idx = train_cont_idx
    )
    trainer.train()

    logger.info(config)
    logger.info(f'finished')


if __name__ == "__main__": main()