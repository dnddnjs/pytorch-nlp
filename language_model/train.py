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
from utils import get_grad_norm, AverageMeter, \
                  download_dataset, batchify, get_batch


parser = argparse.ArgumentParser(description='Character-Aware Neural Language Model')
parser.add_argument('--name', required=True, help='')
parser.add_argument('--gpu', required=True, help='')
parser.add_argument('--dataset', default='penn', help='')
parser.add_argument('--batch_size', default=20, help='')
parser.add_argument('--print_freq', default=50, help='')
parser.add_argument('--num_workers', default=1, help='')
parser.add_argument('--lstm_dim', default=300, help='')
parser.add_argument('--emb_dim', default=15, help='')
parser.add_argument('--seq_length', default=35, help='')
parser.add_argument('--data_path', default='./data/', help='')
parser.add_argument('--logdir', type=str, default='logs/', help='tensorboardx log directory')
args = parser.parse_args()

device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'

torch.manual_seed(7)
np.random.seed(7)


def train_model(epoch):
    model.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    ntokens = len(corpus.dictionary)
    
    hidden = (torch.zeros(2, args.batch_size, args.lstm_dim).to(device),
              torch.zeros(2, args.batch_size, args.lstm_dim).to(device))
    
    for batch, i in enumerate(range(0, train_inputs.size(0) - 1, args.seq_length)):
        data, targets = get_batch(train_inputs, train_targets, i, args)
        data = data.to(device)
        targets = targets.to(device)
        
        model.zero_grad()
        hidden = [state.detach() for state in hidden]
        output, hidden = model(data, hidden)

        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad_norm = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            print('Train Epoch: {} [{}]| Loss: {:.3f} | pexplexity: {:.3f} | grad norm: {:.3f} | batch time: {:.3f}'.format(
                  epoch, batch, losses.val, np.exp(losses.avg), grad_norm, batch_time.avg))    
    
    writer.add_scalar('log/train loss', losses.avg, epoch)
    writer.add_scalar('log/train perplexity', np.exp(losses.avg), epoch)
    
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        
def valid_model(epoch, best_acc, learning_rate):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    ntokens = len(corpus.dictionary)
    hidden = (torch.zeros(2, eval_batch_size, args.lstm_dim).to(device),
              torch.zeros(2, eval_batch_size, args.lstm_dim).to(device))
    
    with torch.no_grad():
        for batch, i in enumerate(range(0, valid_inputs.size(0) - 1, args.seq_length)):
            data, targets = get_batch(valid_inputs, valid_targets, i, args)
            data = data.to(device)
            targets = targets.to(device)

            hidden = [state.detach() for state in hidden]
            output, hidden = model(data, hidden)
            
            loss = F.cross_entropy(output, targets)
            losses.update(loss.item(), args.batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                print('Test Epoch: {} [{}]| Loss: {:.3f} | pexplexity: {:.3f} | batch time: {:.3f}'.format(
                      epoch, batch, losses.avg, np.exp(losses.avg), batch_time.avg))    
                
    # acc = 100.0 * (correct / total)
    writer.add_scalar('log/test loss', losses.avg, epoch)
    writer.add_scalar('log/test perplexity', np.exp(losses.avg), epoch)
    
    if abs(np.exp(losses.avg) - best_acc) < 1 and learning_rate > 0.001:
        learning_rate *= 0.5
        
    if np.exp(losses.avg) < best_acc:
        print('==> Saving model..')
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(model.state_dict(), './save_model/' + args.name + '.pth')
        best_acc = np.exp(losses.avg)

    return best_acc, learning_rate


if __name__ == "__main__":
    start = time.time()
    print('==> download dataset ' + args.dataset)
    download_dataset(args.data_path, args.dataset)
    print('')
    
    print('==> make dataset')
    corpus = Corpus(args.data_path)
    num_chars, num_words = len(corpus.dictionary.idx2char), len(corpus.dictionary.idx2word)
    eval_batch_size = 10
    train_inputs, train_targets = batchify(corpus.train_charidx, corpus.train_wordidx, args.batch_size)
    valid_inputs, valid_targets = batchify(corpus.valid_charidx, corpus.valid_wordidx, eval_batch_size)
    test_inputs, test_targets = batchify(corpus.test_charidx, corpus.test_wordidx, eval_batch_size)
    print('')
    
    print('==> make model')
    model = lstmcnn.LSTMCNN(num_words, num_chars, seq_length=args.seq_length, emb_dim=args.emb_dim, lstm_dim=args.lstm_dim)
    model.to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    print('')
    
    learning_rate = 1.0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    writer = SummaryWriter(args.logdir + args.name)
    
    print('==> train model')
    best_acc = 1000000
    print('----------------------------------------------')
    for epoch in range(35):
        train_model(epoch)
        best_acc, learning_rate = valid_model(epoch, best_acc, learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
        print('best test accuracy is ', best_acc)

    print('overall time which is spent for training is :', time.time() - start)