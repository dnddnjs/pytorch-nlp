import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data as data_utils

import os
import time
import math
import argparse
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from model import lstmcnn
from data import Corpus
from utils import get_grad_norm, AverageMeter, download_dataset, \
                  batchify, get_batch


parser = argparse.ArgumentParser(description='Character-Aware Neural Language Model')
parser.add_argument('--model_path', required=True, help='')
parser.add_argument('--gpu', required=True, help='')
parser.add_argument('--dataset', default='penn', help='')
parser.add_argument('--batch_size', default=20, help='')
parser.add_argument('--print_freq', default=50, help='')
parser.add_argument('--num_workers', default=1, help='')
parser.add_argument('--lstm_dim', default=300, help='')
parser.add_argument('--emb_dim', default=15, help='')
parser.add_argument('--seq_length', default=35, help='')
parser.add_argument('--data_path', default='./data/', help='')
args = parser.parse_args()

device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'

torch.manual_seed(7)
np.random.seed(7)

        
def test_model():
    model.eval()

    losses = AverageMeter()
    
    ntokens = len(corpus.dictionary)
    hidden = (torch.zeros(2, eval_batch_size, args.lstm_dim).to(device),
              torch.zeros(2, eval_batch_size, args.lstm_dim).to(device))
    
    with torch.no_grad():
        for batch, i in enumerate(range(0, test_inputs.size(0) - 1, args.seq_length)):
            data, targets = get_batch(test_inputs, test_targets, i, args)
            data = data.to(device)
            targets = targets.to(device)

            hidden = [state.detach() for state in hidden]
            output, hidden = model(data, hidden)
            
            loss = F.cross_entropy(output, targets)
            losses.update(loss.item(), args.batch_size)

    print('Test Result Loss: {:.3f} | pexplexity: {:.3f}'.format(
           losses.avg, np.exp(losses.avg)))


if __name__ == "__main__":
    print('==> download dataset ' + args.dataset)
    download_dataset(args.data_path, args.dataset)
    print('')
        
    print('==> make dataset')
    corpus = Corpus(args.data_path)
    num_chars, num_words = len(corpus.dictionary.idx2char), len(corpus.dictionary.idx2word)
    eval_batch_size = 10
    print('test data size [ data | label ] : ')
    test_inputs, test_targets = batchify(corpus.test_charidx, corpus.test_wordidx, 
                                         eval_batch_size)
    print('')
        
    print('==> make model')
    model = lstmcnn.LSTMCNN(num_words, num_chars, seq_length=args.seq_length, 
                            emb_dim=args.emb_dim, lstm_dim=args.lstm_dim)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    print('')
    
    print('==> test model')
    test_model()
  