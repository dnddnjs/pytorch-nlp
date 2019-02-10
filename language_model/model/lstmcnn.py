import torch
import torch.nn as nn

# https://github.com/FengZiYjun/CharLM/blob/master/model.py
class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = torch.sigmoid(self.fc1(x))
        return torch.mul(t, torch.relu(self.fc2(x))) + torch.mul(1-t, x)
    
    
class LSTMCNN(nn.Module):
    def __init__(self, num_words, num_chars, seq_length=35, emb_dim=15, lstm_dim=300):
        super(LSTMCNN, self).__init__()
        self.lstm_dim = lstm_dim
        
        kernel_widths = [1, 2, 3, 4, 5, 6]
        lstm_input_size = int(sum(kernel_widths) * 25)
        self.char_embed = nn.Embedding(num_chars, embedding_dim=emb_dim, padding_idx=0)
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for kernel_width in kernel_widths:
            conv_layer = nn.Conv1d(emb_dim, int(kernel_width*25), kernel_size=kernel_width, padding=0)
            pool_layer = nn.MaxPool1d(19-kernel_width+1)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(pool_layer)
        
        self.highway = Highway(input_size=lstm_input_size)
        self.lstm = nn.LSTM(lstm_input_size, lstm_dim, num_layers=2)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(lstm_dim, num_words)
        
        for param in self.parameters():
            nn.init.uniform_(param.data, -0.05, 0.05)
        
    def forward(self, x, hidden):
        # [20, 35, 19]
        batch_size, seq_len, _ = x.size()
        x = x.contiguous()
        x = x.view(batch_size*seq_len, x.size(2))
        x = self.char_embed(x)
        x = x.transpose(1, 2)
        # [700, 15, 19]

        conv_outs = []
        for i in range(len(self.conv_layers)):
            conv_layer = self.conv_layers[i]
            pool_layer = self.pool_layers[i]
            conv_out = conv_layer(x)
            conv_out = torch.tanh(conv_out)
            conv_out = pool_layer(conv_out)
            conv_out = conv_out.squeeze(-1)
            conv_outs.append(conv_out)
        
        x = torch.cat(conv_outs, dim=1)
        # [700, 525]
        x = self.highway(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        # [35, 20, 525]

        output, hidden = self.lstm(x, hidden)
        # [35, 20, 300]
        x = self.dropout(output)
        x = x.transpose(0, 1).contiguous()
        x = x.view(x.size(0)*x.size(1), x.size(2))
        out = self.fc_out(x)
        return out, hidden