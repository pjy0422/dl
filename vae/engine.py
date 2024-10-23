import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from custom_dataloader import CustomDatasetLoader
from model import VAE
from torchvision.utils import save_image


def train(model, train_loader, optimizer, config: dict):
    epochs = config["epochs"]
    device = config["device"]
    for epoch in range(epochs):
        model.train()
        train_loss = 0  # Reset at the start of each epoch
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kld_loss = model.loss_function(
                recon_batch, data, mu, logvar
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % config.get("log_interval", 100) == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\t"
                    f"Reconstruction Loss: {recon_loss.item():.6f}\t"
                    f"KLD Loss: {kld_loss.item():.6f}"
                )

        average_loss = train_loss / len(train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {average_loss:.4f}")


def val(model, val_loader, config: dict):
    device = config["device"]
    val_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += model.loss_function(recon_batch, data, mu, logvar)[0]
            if i == 0:
                n = min(data.size(0), 32)
                comparison = torch.cat(
                    [
                        data[:n],
                        recon_batch.view(
                            config["dataset"]["batch_size"], 1, 28, 28
                        )[:n],
                    ]
                )
                os.makedirs("results", exist_ok=True)
                save_image(
                    comparison.cpu(),
                    "results/reconstruction_" + str(10) + ".png",
                    nrow=n,
                )

    print(
        f"====> Validation set loss: {val_loss / len(val_loader.dataset):.4f}"
    )


def main():
    data = CustomDatasetLoader("config.yaml")
    train_loader = data.train_loader
    val_loader = data.val_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open("config.yaml"))
    model = VAE(latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, optimizer, config)
    val(model, val_loader, config)


if __name__ == "__main__":
    main()
