import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data as data_utils

import os
import time
import argparse
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from model import CharCNN
from dataset import TextDataset
from utils import get_grad_norm, AverageMeter


parser = argparse.ArgumentParser(description='Character-level convolutional neural network for text classification')
parser.add_argument('--batch_size', default=512, help='')
parser.add_argument('--print_freq', default=30, help='')
parser.add_argument('--num_workers', default=8, help='')
parser.add_argument('--num_feature', default=1024, help='')
parser.add_argument('--num_channels', default=256, help='')
parser.add_argument('--large_model', default=False, help='')
parser.add_argument('--seq_length', default=1014, help='')
parser.add_argument('--data_path', default='./data/', help='')
parser.add_argument('--logdir', type=str, default='logs', help='tensorboardx log directory')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(7)
np.random.seed(7)


def train_model(epoch):
    model.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):        
        if inputs.size(0) < args.batch_size:
            continue
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.long().squeeze(-1)
        

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(model)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{}]| Loss: {:.3f} | acc: {:.3f} | grad norm: {:.3f} | batch time: {:.3f}'.format(
                  epoch,  batch_idx, len(train_loader), losses.val, correct / total, grad_norm, batch_time.avg))    
    
    acc = 100.0 * (correct / total)
    writer.add_scalar('log/train accuracy', acc, epoch)
    writer.add_scalar('log/train loss', losses.avg, epoch)
    
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        
def test_model(epoch, best_acc):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if inputs.size(0) < args.batch_size:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long().squeeze(-1)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            losses.update(loss.item(), args.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('Test Epoch: {} [{}/{}]| Loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}'.format(
                      epoch,  batch_idx, len(test_loader), losses.val, correct / total, batch_time.avg))    
                
    acc = 100.0 * (correct / total)
    writer.add_scalar('log/test accuracy', acc, epoch)
    writer.add_scalar('log/test loss', losses.avg, epoch)
    
    if acc > best_acc:
        print('==> Saving model..')
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(model.state_dict(), './save_model/ckpt.pth')
        best_acc = acc

    return best_acc


if __name__ == "__main__":
    start = time.time()
    vocab_list = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    print('==> convert csv to hdf5')
    # convert_csv_hdf5(args.data_path, args.seq_length, vocab_list)
        
    print('==> make dataset')
    train_dataset = TextDataset(args.data_path, args.seq_length, vocab_list, is_train=True)
    test_dataset = TextDataset(args.data_path, args.seq_length, vocab_list, is_train=False)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, 
                                         shuffle=True, num_workers=args.num_workers)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, 
                                        shuffle=True, num_workers=args.num_workers)
    print(train_dataset.num_classes)
    asdf
    print('==> make model')
    model = CharCNN(num_classes=train_dataset.num_classes, seq_length=1014, in_channels=68, 
                    num_channels=256, num_features=1024, large=args.large_model)
    model.to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    writer = SummaryWriter(args.logdir)
    
    best_acc = 0
    print('----------------------------------------------')
    for epoch in range(20):
        train_model(epoch)
        best_acc = test_model(epoch, best_acc)
        print('best test accuracy is ', best_acc)

    print('overall time which is spent for training is :', time.time() - start)