from __future__ import print_function
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DAE(nn.Module):
    def __init__(self, in_channel=10240, feature_size=32, out_channel=2):        
        super(DAE, self).__init__()
        self.in_channel = in_channel
        
        self.en_fc1 = nn.Sequential(
            nn.Linear(in_channel, 2046),
            nn.BatchNorm1d(2046),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))

        self.en_fc2 = nn.Sequential(
            nn.Linear(2046, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))
    
        self.en_fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))

        self.en_fc4 = nn.Sequential(
            nn.Linear(256, feature_size),)
        
        self.de_fc1 = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))

        self.de_fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))

        self.de_fc3 = nn.Sequential(
            nn.Linear(512, 2046),
            nn.BatchNorm1d(2046),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))

        self.de_fc4 = nn.Sequential(
            nn.Linear(2046, in_channel),)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size, out_channel))

    def forward(self, x):
        noise = (torch.rand(x.shape).to(device) * x.mean().to(device) / 10).to(device)
        x = x + noise
        out = x.view(x.size(0), -1)
        out = self.en_fc1(out)
        out = self.en_fc2(out)
        out = self.en_fc3(out)
        encode = self.en_fc4(out)

        out = self.de_fc1(encode)
        out = self.de_fc2(out)
        out = self.de_fc3(out)
        decoded = self.de_fc4(out)

        label = self.classifier(encode)
        return decoded, label
