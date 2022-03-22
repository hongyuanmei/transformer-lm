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
from tester import Tester
from tokenizer import Tokenizer
from dataset import Dataset
from model import TransformerLM
from utils import TqdmLoggingHandler, get_rawdata

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=32, type=int,
        help='valid/test batch size')
    parser.add_argument('--bptt', default=512, type=int, 
        help='len of seq in each batch')
    parser.add_argument('--dataset', default='wikitext-2', type=str, 
        help='dataset')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'], 
        help='train/valid/test split')
    parser.add_argument('--pretrained', default='', type=str, 
        help='pretrained model')
    parser.add_argument('--n_layers', default=8, type=int, 
        help='num of layers')
    parser.add_argument('--d_model', default=512, type=int, 
        help='dimension of model hidden state')
    parser.add_argument('--n_heads', default=4, type=int, 
        help='num of heads')
    parser.add_argument('--d_ffn', default=1024, type=int, 
        help='dimension of feedforward network')
    parser.add_argument('--ada', default=1, type=int, choices=[0,1], 
        help='adaptive input and softmax: 0=no, 1=yes')
    parser.add_argument('--adacutoff', default='20000,60000', type=str,  
        help='cutoff list for adaptive input and softmax e.g., 20000,60000')
    parser.add_argument('--tie_emb', default=1, type=int, choices=[0,1], 
        help='tie input and output embeddings: 0=no, 1=yes')
    parser.add_argument('--dropout', default=0.0, type=float,
        help='dropout rate')
    parser.add_argument('--seed', default=12345, type=int, 
        help='random seed')
    parser.add_argument('--root_path', default='..', type=str, 
		help='root path of project')

    args = parser.parse_args()

    # tokenize data 
    rawdata_test = get_rawdata(args.dataset, args.split, args.root_path)

    # choose dictionary mode based on data
    if args.dataset == 'wikitext-103': 
        tokenizer = Tokenizer(
            mode='ext_dict', 
            extdict=os.path.join(args.root_path, f'data/{args.dataset}/dict.txt')
        )
    elif args.dataset == 'wikitext-2': 
        tokenizer = Tokenizer(
            mode='ext_dict', 
            extdict=os.path.join(args.root_path, f'data/{args.dataset}/dict.txt')
        )
    else: 
        tokenizer = Tokenizer(
            mode='use_all', 
            data=rawdata_train
        )

    # Dataset
    data_test = Dataset(
        tokenizer.tokenize_data(rawdata_test), 
        args.batch_size, args.bptt, tokenizer.get_pad())
    
    # set random numbers 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    model = TransformerLM(
        tokenizer.get_size(), 
        args.n_layers, args.d_model, args.n_heads, args.d_ffn, 
        args.ada, sorted([int(x) for x in args.adacutoff.split(',')]), 
        tokenizer.get_pad(), 
        args.tie_emb, args.dropout
    )
    #modelfile = os.path.join(logfolder, f'model')

    # read saved model
    model_state = torch.load(
        os.path.join(
            args.root_path, f"pretrained/{args.dataset}/{args.pretrained}"))
    model.load_state_dict(model_state)
    # move to GPU
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_m = round(total_params / 1000000.0)
    print(f"# of model parameters : {total_params} or {total_params_in_m}M")
    #if torch.cuda.is_available():
    #    model = model.to('cuda')
    
    # loss 
    criterion = NLL(tokenizer.get_pad())

    # Run
    tester = Tester(
        args = args, model = model, criterion = criterion, data_loader = data_test)
    tester.eval()

    print(f'finished')


if __name__ == "__main__": main()