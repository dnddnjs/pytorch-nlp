import torch
import h5py
import os


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)
    return grad_norm

    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def download_dataset(data_path, dataset):
    if not os.path.isdir(data_path):
        os.system('mkdir -p {}'.format(data_path))
        if dataset == 'amazon':
            os.system('gdown https://drive.google.com/uc?id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA -O {}/dataset.tar.gz'.format(data_path))
            os.system('tar xf {}/dataset.tar.gz -C {} --strip-components=1'.format(data_path, data_path))

        elif dataset == 'penn':
            os.system('wget -O data/train.txt https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt')
            os.system('wget -O data/valid.txt https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt')
            os.system('wget -O data/test.txt https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt')
    os.system('ls -lh {}'.format(data_path))
        

def batchify(char_data, word_data, batch_size): 
    nbatch = char_data.size(0) // batch_size
    char_data = char_data.narrow(0, 0, nbatch * batch_size)
    word_data = word_data.narrow(0, 0, nbatch * batch_size)
    char_data = char_data.view(batch_size, nbatch, char_data.size(-1)).transpose(0, 1).contiguous()
    word_data = word_data.view(batch_size, -1).t().contiguous()
    print(list(char_data.size()), list(word_data.size()))
    return char_data, word_data


def get_batch(char_data, word_data, i, args):
    seq_len = min(args.seq_length, len(char_data) - 1 - i)
    data = char_data[i:i+seq_len]
    target = word_data[i+1:i+1+seq_len]
    return data.transpose(0, 1), target.transpose(0, 1).contiguous().view(-1)