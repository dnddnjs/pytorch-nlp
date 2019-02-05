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


# code from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
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
        

def convert_csv_hdf5(data_path, seq_length, vocal_list):
    num_train_data = 0
    num_test_data = 0
    
    # training data
    file_path = data_path + '/train.h5'
    if os.path.isfile(file_path):
        h5f = h5py.File(file_path, 'r')
        data_string = h5f['train']
        num_train_data = data_string.shape[0]
        print('the number of train data is ', num_train_data)
    else:
        data_list = []
        path = data_path + '/train.csv'
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                
                data_list.append(line)

        num_train_data = len(data_list)
        print('the number of train data is ', num_train_data)
        
        f = h5py.File(file_path, 'w')
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset("train", (len(data_list),), dtype=dt)

        for i, data in enumerate(data_list):
            if i % 10000 == 0:
                print(i)
            dset[i] = data_list[i]


    # training data
    file_path = data_path + '/test.h5'
    if os.path.isfile(file_path):
        h5f = h5py.File(file_path, 'r')
        data_string = h5f['test']
        num_train_data = data_string.shape[0]
        print('the number of train data is ', num_train_data)
    else:
        data_list = []
        path = data_path + '/test.csv'
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data_list.append(line)

        num_train_data = len(data_list)
        print('the number of test data is ', num_train_data)
        
        f = h5py.File(file_path, 'w')
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset("test", (len(data_list),), dtype=dt)

        for i, data in enumerate(data_list):
            if i % 10000 == 0:
                print(i)
            dset[i] = data_list[i]
            
    return num_train_data, num_test_data


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
        
        