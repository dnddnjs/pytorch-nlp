import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, num_classes=14, seq_length=1014, in_channels=68, num_channels=256, 
                 num_features=1024, large=True):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_channels, kernel_size=7, padding=0)
        self.maxpool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=0)
        self.maxpool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.conv5 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.conv6 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.maxpool6 = nn.MaxPool1d(3)

        num_conv_out = int((seq_length - 96) / 27 * num_channels)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_conv_out, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, num_classes)

        if not large:
            self.init_weight(std=0.05)
        else:
            self.init_weight(std=0.02)

    def init_weight(self, mean=0.0, std=0.05):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(mean, std)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.maxpool6(torch.relu(self.conv6(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout((self.fc1(x)))
        x = self.dropout((self.fc2(x)))
        out = self.fc3(x)
        return out
    
    
class CharCNNBN(nn.Module):
    def __init__(self, num_classes=14, seq_length=1014, in_channels=68, num_channels=256, 
                 num_features=1024, large=True):
        super(CharCNNBN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_channels, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.maxpool1 = nn.MaxPool1d(3)
        
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=0)
        self.maxpool2 = nn.MaxPool1d(3)
        self.bn2 = nn.BatchNorm1d(num_channels)
        
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm1d(num_channels)
        
        self.conv4 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm1d(num_channels)
        
        self.conv5 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm1d(num_channels)
        
        self.conv6 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn6 = nn.BatchNorm1d(num_channels)
        self.maxpool6 = nn.MaxPool1d(3)

        num_conv_out = int((seq_length - 96) / 27 * num_channels)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_conv_out, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, num_classes)

        if not large:
            self.init_weight(std=0.05)
        else:
            self.init_weight(std=0.02)

    def init_weight(self, mean=0.0, std=0.05):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(mean, std)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.maxpool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.maxpool6(torch.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        out = self.fc3(x)
        return out