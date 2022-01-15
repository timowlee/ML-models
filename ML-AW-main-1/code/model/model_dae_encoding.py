from __future__ import print_function
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DAE_Encoder(nn.Module):
    def __init__(self, in_channel=10240, feature_size=32):
        super(DAE_Encoder, self).__init__()
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

    def forward(self, x):
        noise = (torch.rand(x.shape).to(device) * x.mean().to(device) / 10).to(device)
        x = x + noise
        out = x.view(x.size(0), -1)
        out = self.en_fc1(out)
        out = self.en_fc2(out)
        out = self.en_fc3(out)
        encode = self.en_fc4(out)
        return encode

class DAE_Decoder(nn.Module):
    def __init__(self, in_channel=10240, feature_size=32):        
        super(DAE_Decoder, self).__init__()
        self.in_channel = in_channel
    
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
            
    def forward(self, x):
        out = self.de_fc1(x)
        out = self.de_fc2(out)
        out = self.de_fc3(out)
        decoded = self.de_fc4(out)
        return decoded

class DAE_Encoding(nn.Module):
    def __init__(self, in_channel=10240, feature_size=512, out_channel=2):        
        super(DAE_Encoding, self).__init__()

        self.kernel_num = 16
        self.V = 25
        self.embedding_dim = 800

        self.encoder = DAE_Encoder(in_channel, feature_size)
        self.decoder = DAE_Decoder(in_channel, feature_size)

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, out_channel))

    def forward(self, x):
        encode = self.encoder.forward(x)
        decode = self.decoder.forward(encode)
        class_output = self.classifier(encode)
        return decode, class_output

class DAE_Encoding_Conv(nn.Module):
    def __init__(self, in_channel=10240, feature_size=512, out_channel=2):        
        super(DAE_Encoding_Conv, self).__init__()

        self.kernel_num = 16
        self.V = 25
        self.embedding_dim = 800

        self.encoder = DAE_Encoder(in_channel, feature_size)
        self.decoder = DAE_Decoder(in_channel,feature_size)

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

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, out_channel))

    def forward(self, x):
        encode = self.encoder.forward(x)
        decode = self.decoder.forward(encode)
        encode = encode.unsqueeze(1)
        encode = self.embed1(encode)
        encode = self.embed2(encode)
        encode = encode.view(encode.size(0),-1)
        class_output = self.classifier(encode)
        return decode, class_output
