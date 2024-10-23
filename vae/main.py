import torch.optim as optim
import yaml
from custom_dataloader import CustomDatasetLoader
from engine import train, val
from model import VAE


def main():

    config = yaml.safe_load(open("config.yaml"))

    data = CustomDatasetLoader("config.yaml")
    train_loader = data.train_loader
    val_loader = data.val_loader
    device = config["model"]["device"]
    latent_dim = config["model"]["latent_dim"]
    lr = config["model"]["lr"]
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, config)
    val(model, val_loader, config)


if __name__ == "__main__":
    main()
