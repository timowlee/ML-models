import torch.nn.functional as F
from torch import nn
import torch
import math


def nll_loss(output, target):
    return F.nll_loss(output, target)

def crossentropy_loss(output, target):
    loss = nn.CrossEntropyLoss()
    return loss(output, target)

def MSE_loss(output, target):
    loss = nn.MSELoss()
    return loss(output, target)

def DIAE_loss(x_act, x_pred, y_act, y_pred, l_loss, w_loss, 
              alpha=1.0, beta=1.0, gamma=1.0):
    """
    Def:
        DIAE_loss(x_act, x_pred, y_act, y_pred, l_loss, w_loss, 
                      alpha=1.0, beta=1.0, gamma=1.0)
        calculates overall loss function for DIAE model mentioned in
        https://doi.org/10.1016/j.ymssp.2020.107233
        research paper.
    Args:
        x_act:  Actual input to be reconstructed (array)
        x_pred: Reconstructed input (array)
        y_act:  Ground truth labels for output (array)
        y_pred: Predicted output (array)
        l_loss: Loss for matrix L calculated by reduced sum of (L-Lt)^2 (float)
        w_loss: Loss for reconstruction and prediction weight matrices 
                from encoder, decoder and predictor calculates as reduced
                sum of (We^2 + Wd^2 + Wp^2). We, Wd and Wp are representations
                for weight matrices of encoder, decoder and predictor models
                (float)
        alpha:  Regularization parameter for reconstruction loss.
                Default:1.0 (float)
        beta:   Regularization parameter for l_loss. Default:1.0 (float)
        gamma:  Regularization parameter for w_loss. Default:1.0 (float)
    Returns:
        j:      Overall loss function for DIAE
    """
    
    def gaussian_kernel(y, y_hat):
        y_dif = y - y_hat
        std_2 = torch.square(torch.std(y_dif))
        g = 1/((2*math.pi*std_2) * torch.exp(-torch.square(y_dif)/(2*std_2)))
        return torch.mean(g)
    
    # Objective function for discriminant information of output prediction y using Gaussian Kernel   
    j1 = gaussian_kernel(y_act, y_pred)
    # Reconstruction loss for input x which is the same as classical autoencoder
    j2 = (alpha/2.0) * torch.mean(torch.square(x_act - x_pred))
    # Objective function for degree of relatedness to keep matricx L symmetric
    j3 = (beta/2.0) * l_loss
    # Frobenius norm of weights for reconstruction and prediction of x and y respectively
    j4 = (gamma/2.0) * w_loss
    
    # Combined objective function for DIAE model
    j = j1 + j2 + j3 + j4
    
    return j