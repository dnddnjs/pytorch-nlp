import torch
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/dreamgonfly/deep-text-classification-pytorch/blob/master/models/VDCNN.py
class KMaxPool(nn.Module):
    def __init__(self, k='half'):
        super(KMaxPool, self).__init__()
        self.k = k
        
    def forward(self, x):
        kmax, kargmax = x.topk(self.k, dim=2)
        return kmax

	
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if down_sample:
            self.down_sample = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                         stride=2)
            self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.down_sample = None
            self.max_pool = None


    def forward(self, x):
        shortcut = x
        
        if self.max_pool is not None:
            x = self.max_pool(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(shortcut)

        out += shortcut
        out = torch.relu(out)
        return out


class DeepCNN(nn.Module):
    def __init__(self, num_chars, num_layers, block, num_classes=10):
        super(DeepCNN, self).__init__()
        self.num_layers = num_layers
        self.char_embed = nn.Embedding(num_chars+1, embedding_dim=16, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.layers_2n = self.get_layers(block, 64, 64, self.num_layers[0], False)
        self.layers_4n = self.get_layers(block, 64, 128, self.num_layers[1], True)
        self.layers_6n = self.get_layers(block, 128, 256, self.num_layers[2], True)
        self.layers_8n = self.get_layers(block, 256, 512, self.num_layers[3], True)

        self.kmax_pool = KMaxPool(k=8)
        # self.avg_pool = nn.AvgPool1d(8, stride=1)
        self.fc1 = nn.Linear(512*8, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, in_channels, out_channels, num_layers, down_sample):
        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, down_sample)])

        for _ in range(num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.char_embed(x)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)
        x = self.layers_8n(x)
        
        x = self.kmax_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def deepcnn():
    block = ResidualBlock
    model = DeepCNN(69, [2, 2, 2, 2], block, num_classes=5) 
    return model