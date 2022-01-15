from __future__ import print_function
import torch.nn as nn
import torch
import pyro.distributions as dist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VAE_Encoder(nn.Module):
    def __init__(self, z_dim=256, hidden_dim=1024, in_channel=10240, feature_size=32):
        super(VAE_Encoder, self).__init__()
        self.in_channel = in_channel

        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):

        out = x.view(x.size(0), -1)

        #x = x.reshape(-1, 784)

        hidden = self.softplus(self.fc1(out))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale

class VAE_Decoder(nn.Module):
    def __init__(self, z_dim=256, hidden_dim=1024, in_channel=10240, feature_size=32):

        super(VAE_Decoder, self).__init__()
        self.in_channel = in_channel
    
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 10240)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))

        return loc_img

class VAE_Encoding(nn.Module):
    def __init__(self, z_dim = 256, hidden_dim = 1024, in_channel=10240, out_channel=2):        
        super(VAE_Encoding, self).__init__()

        self.encoder = VAE_Encoder(z_dim = z_dim, hidden_dim = hidden_dim)
        self.decoder = VAE_Decoder(z_dim = z_dim, hidden_dim = hidden_dim)

        self.z_dim = z_dim

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, out_channel))

    def forward(self, x):

        z_loc, z_scale = self.encoder(x)
        encode = dist.Normal(z_loc, z_scale).sample()
        decode = self.decoder(encode)

        class_output = self.classifier(encode)
        return decode, class_output

