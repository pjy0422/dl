import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

latent_dim = 30


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Latent space layers
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Encode the input image to obtain the latent mean (mu) and log variance (logvar).
        Args:
            x: Input tensor of shape [B, H, W, C]
        Returns:
            mu, logvar: Tensors representing mean and log variance of latent space
        """
        x = x.view(-1, 784)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize to sample from the latent space.
        Args:
            mu: Mean of the latent space
            logvar: Log variance of the latent space
        Returns:
            z: Latent variable sampled from the distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the latent variable to reconstruct the image.
        Args:
            z: Latent variable
        Returns:
            x_recon: Reconstructed image of shape [B, 1, 28, 28]
        """
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        """
        Forward pass through the VAE.
        Args:
            x: Input tensor
        Returns:
            x_recon: Reconstructed image
            mu: Latent mean
            logvar: Latent log variance
            loss: Total loss (reconstruction + KLD)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute the loss function for the VAE.
        Args:
            recon_x: Reconstructed image
            x: Input image
            mu: Latent mean
            logvar: Latent log variance
        Returns:
            loss: Total loss (reconstruction + KLD)
            reconstruct_loss: Reconstruction loss component
            kld_loss: Kullback-Leibler divergence loss component
        """
        batch_size = x.size(0)
        # Reconstruction loss
        reconstruct_loss = (
            F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
        )

        # Kullback-Leibler divergence
        kld_loss = (
            -0.5
            * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            / batch_size
        )

        # Total loss
        loss = reconstruct_loss + kld_loss
        return loss, reconstruct_loss, kld_loss
