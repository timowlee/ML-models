from __future__ import print_function
import torch.nn as nn
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    Encoder layers for getting hidden representation from input 
    """
    def __init__(self, z_dim=256, hidden_dim=1024, in_channel=10240):
        super(Encoder, self).__init__()
        self.W1 = nn.Linear(in_channel, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        
        out = torch.sigmoid(self.W1(out))
        out = self.W2(out)
        
        # Adding squares of all weights for normalization
        j_e = (torch.sum(torch.square(self.W1.weight)) + 
                torch.sum(torch.square(self.W2.weight)))
        
        return out, j_e
        
class Decoder(nn.Module):
    """
    Decoder layers for reconstructing input from hidden representation from
    encoder layers
    """
    def __init__(self, z_dim=256, hidden_dim=1024, out_channel=10240):
        super(Decoder, self).__init__()
        self.W1 = nn.Linear(z_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_channel)

    def forward(self, z):
        z = torch.sigmoid(self.W1(z))
        z = self.W2(z)
        
        # Adding squares of all weights for normalization
        j_d = (torch.sum(torch.square(self.W1.weight)) + 
                torch.sum(torch.square(self.W2.weight)))
        
        return z, j_d

class Predictor(nn.Module):
    """
    Predictor layers for predicting output y
    """
    def __init__(self, z_dim=256, hidden_dim=32, out_channel=1):
        super(Predictor, self).__init__()
        self.W1 = nn.Linear(z_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_channel)
    
    def forward(self, z):
        z = torch.relu(self.W1(z))
        z = torch.sigmoid(self.W2(z))
        
        # Adding squares of all weights for normalization
        j_p = (torch.sum(torch.square(self.W1.weight)) + 
                torch.sum(torch.square(self.W2.weight)))
        
        return z, j_p

class  Discriminant(nn.Module):
    """
    Discriminant layer to add the related information matrix L 
    """
    def __init__(self, out_channel=1):
        super(Discriminant, self).__init__()
        # Discriminant layer only has weight matrix and there is no bias
        self.L = nn.Linear(out_channel, out_channel, bias=False)
    
    def forward(self, z):
        out = self.L(z)
        
        # Loss for keeping weight L symmetric
        j_l = torch.sum(torch.square(self.L.weight - self.L.weight.t()))
        
        return out, j_l

class DIAE(nn.Module):
    """
    DIAE model according to the following research paper
    https://doi.org/10.1016/j.ymssp.2020.107233
    """
    def __init__(self, in_channel=10240, hidden_dim=1024, z_dim=256,
                 prediction_hidden_dim=32,out_channel=1):
        super(DIAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.predictor = Predictor()
        self.discriminant = Discriminant()
    
    def forward(self, x):
        z_e, j_e = self.encoder(x)
        x_hat, j_d = self.decoder(z_e)
        y_hat, j_p = self.predictor(z_e)
        y_pred, j_l = self.discriminant(y_hat)
        
        # Term j_e + j_d + j_p is the accumulative normalization loss 
        # for all weights of encoder, decoder and predictor
        return x_hat, y_pred, j_e + j_d + j_p, j_l
        