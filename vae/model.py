import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    """
    VAE encoder class
    """

    pass


class Decoder(nn.Module):
    """
    VAE decoder class
    """

    pass


class VAE(nn.Module):
    def __init__(self, config: dict):
        super(VAE, self).__init__()
        self.config = config
