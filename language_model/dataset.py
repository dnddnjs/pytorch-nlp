import os
import csv
from time import time
import torch
import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data_path, seq_length, vocab_list, is_train):
        self.data_list = []
        self.label_list = []
        self.seq_length = seq_length
        self.vocab_list = vocab_list
        
        self.identity_mat = np.identity(len(self.vocab_list))
        
        if is_train:
            path = data_path + '/train.txt'
        else:
            path = data_path + '/valid.txt'
            
        with open(path, encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for line in reader:
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                
                label = int(line[0]) - 1
                self.label_list.append(label)
                self.data_list.append(text.lower())
        
        self.num_classes = len(set(self.label_list))
        self.len = len(self.data_list)
        
    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]

        char_id_list = []
        char_feature_list = []
        
        for char in data:
            try:
                char_id = self.vocab_list.index(char)
                char_feature = self.identity_mat[char_id]
                char_id += 1
            except:
                char_feature = np.zeros(len(self.vocab_list))
                char_id = 0
            char_id_list.append(char_id)
            char_feature_list.append(char_feature)

        if len(char_feature_list) > self.seq_length:
            char_feature_list = char_feature_list[:self.seq_length]
            char_id_list = char_id_list[:self.seq_length]
            
        if len(char_feature_list) < self.seq_length:
            for i in range(self.seq_length-len(char_feature_list)):
                char_feature = np.zeros(len(self.vocab_list))
                char_feature_list.append(char_feature)
                char_id_list.append(0)
        
        data = torch.Tensor(np.stack(char_feature_list))
        data_id = torch.Tensor(np.stack(char_id_list)).long()
        label = torch.Tensor([label])
        return data, data_id, label
 
    def __len__(self):
        return len(self.data_list)