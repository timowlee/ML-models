from __future__ import print_function
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Conv_DAE(nn.Module):

    def __init__(self, in_feature=10240, feature_size=32, out_channel=2):        
        super(Conv_DAE, self).__init__()
        self.in_channel = 1
        
        self.kernel_num = 16
        self.V = 25

        self.embed1 = nn.Sequential(
            nn.Conv1d(1, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.embed2 = nn.Sequential(
            nn.Conv1d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(self.V))
        
        self.en_fc1 = nn.Sequential(
            nn.Linear(800, 2046),
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
            nn.Linear(2046, in_feature),)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size, out_channel))

    def forward(self, x):
        noise = (torch.rand(x.shape).to(device) * x.mean().to(device) / 10).to(device)
        x = x + noise
        x = x.unsqueeze(1)
        x = self.embed1(x)
        x = self.embed2(x)
        pre_encoded = x.view(x.size(0),-1)
        out = self.en_fc1(pre_encoded)
        out = self.en_fc2(out)
        out = self.en_fc3(out)
        encode = self.en_fc4(out)

        out = self.de_fc1(encode)
        out = self.de_fc2(out)
        out = self.de_fc3(out)
        decoded = self.de_fc4(out)

        label = self.classifier(encode)
        return decoded, label


class Shallow_Conv_DAE(nn.Module):
    def __init__(self, in_feature=10240, feature_size=32, out_channel=2):        
        super(Shallow_Conv_DAE, self).__init__()
        self.in_channel = 1
        
        self.kernel_num = 16
        self.V = 25

        self.embed1 = nn.Sequential(
            nn.Conv1d(1, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.embed2 = nn.Sequential(
            nn.Conv1d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(self.V))

        self.en_fc1 = nn.Sequential(
            nn.Linear(800, feature_size),)

        self.de_fc1 = nn.Sequential(
            nn.Linear(feature_size, in_feature),)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size, out_channel))

    def forward(self, x):
        noise = (torch.rand(x.shape).to(device) * x.mean().to(device) / 10).to(device)
        x = x + noise
        x = x.unsqueeze(1)
        x = self.embed1(x)
        x = self.embed2(x)
        pre_encoded = x.view(x.size(0),-1)
        encode = self.en_fc1(pre_encoded)

        decoded = self.de_fc1(encode)

        label = self.classifier(encode)
        return decoded, label


class Shallowest_Conv_DAE(nn.Module):
    def __init__(self, in_feature=10240, feature_size=32, out_channel=2):        
        super(Shallowest_Conv_DAE, self).__init__()
        self.in_channel = 1
        
        self.kernel_num = 16
        self.V = 25

        self.embed1 = nn.Sequential(
            nn.Conv1d(1, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.en_fc1 = nn.Sequential(
            nn.Linear(81920, feature_size),)

        self.de_fc1 = nn.Sequential(
            nn.Linear(feature_size, in_feature),)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size, out_channel))

    def forward(self, x):
        noise = (torch.rand(x.shape).to(device) * x.mean().to(device) / 10).to(device)
        x = x + noise
        x = x.unsqueeze(1)
        x = self.embed1(x)
        pre_encoded = x.view(x.size(0),-1)

        encode = self.en_fc1(pre_encoded)

        decoded = self.de_fc1(encode)

        label = self.classifier(encode)
        return decoded, label
