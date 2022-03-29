import torch
import torch.nn as nn

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import pickle 
import os

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        # encoder produces mean and log of variance
        # (i.e., parateters of simple tractable normal distribution "q"
        log_var = self.FC_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.tanh(self.FC_output(h))
        return x_hat

class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim, DEVICE):
        super(VAE, self).__init__()
        self.Encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
        self.DEVICE = DEVICE

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon
        z = mean + var*epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        # takes exponential function (log var -> var)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var