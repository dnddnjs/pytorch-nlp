import os
from io import open
import torch

# code from https://github.com/pytorch/examples/blob/master/word_language_model/data.py
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.char2idx = {'PAD': 0}
        self.idx2char = ['PAD']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]
    
    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        train_path = os.path.join(path, 'train.txt')
        valid_path = os.path.join(path, 'valid.txt')
        test_path = os.path.join(path, 'test.txt')
        
        train_size = self.make_dictionary(train_path)
        valid_size = self.make_dictionary(valid_path)
        test_size = self.make_dictionary(test_path)
        
        print('character dictionary')
        print(self.dictionary.char2idx)
        
        self.max_word_len = max([len(word) for word in self.dictionary.word2idx])

        self.train_charidx, self.train_wordidx = self.tokenize(train_path, train_size)
        self.valid_charidx, self.valid_wordidx = self.tokenize(valid_path, valid_size)
        self.test_charidx, self.test_wordidx = self.tokenize(test_path, test_size)

    def make_dictionary(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                    for char in word:
                        self.dictionary.add_char(char)
        return tokens
                        
    def tokenize(self, path, data_size):
        with open(path, 'r', encoding="utf8") as f:
            word_indexs = torch.LongTensor(data_size)
            char_indexs = torch.LongTensor(data_size, self.max_word_len)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    word_index = self.dictionary.word2idx[word]
                    word_indexs[token] = word_index
                    
                    char_list = []
                    chars = 0
                    for char in word:
                        char_idx = self.dictionary.char2idx[char]
                        char_indexs[token, chars] = char_idx
                        chars += 1
                                       
                    token += 1
                    
        return char_indexs, word_indexs