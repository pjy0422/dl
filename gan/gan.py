import os

import torch
import torch.nn as nn
from maxout import MaxOut
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, output_dim: int = 28 * 28
    ) -> None:
        super(Generator, self).__init__()
        self.model = self._build_model(hidden_dim, output_dim)

    @staticmethod
    def _build_model(hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        latent_dim: int = 256,
        dropout: float = 0.2,
        num_pieces: int = 5,
    ) -> None:
        super(Discriminator, self).__init__()
        self.model = self._build_model(
            input_dim, latent_dim, dropout, num_pieces
        )

    @staticmethod
    def _build_model(
        input_dim: int, latent_dim: int, dropout: float, num_pieces: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 2 * latent_dim),
            MaxOut(input_dim=2 * latent_dim, num_pieces=num_pieces),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
            MaxOut(input_dim=latent_dim, num_pieces=num_pieces),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GAN(nn.Module):
    def __init__(self, config: dict) -> None:
        super(GAN, self).__init__()
        self.config = config
        output_dim = self._calc_output_dim()
        self.generator = Generator(
            hidden_dim=config["generator"]["hidden_dim"], output_dim=output_dim
        )
        self.discriminator = Discriminator(
            input_dim=output_dim,
            latent_dim=config["discriminator"]["latent_dim"],
            dropout=config["discriminator"]["dropout"],
            num_pieces=config["discriminator"]["num_pieces"],
        )

    def _calc_output_dim(self) -> int:
        dataset_config = self.config["dataset"]
        return (
            dataset_config["width"]
            * dataset_config["height"]
            * dataset_config["channels"]
        )


class GANTrainer:
    def __init__(
        self,
        model: GAN,
        config: dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.device = config["trainer"]["device"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = config["trainer"]["batch_size"]

        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

        self.optimizer_g = self._init_optimizer(
            self.model.generator, "generator"
        )
        self.optimizer_d = self._init_optimizer(
            self.model.discriminator, "discriminator"
        )

    def _init_optimizer(self, model, model_type: str) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.config[model_type]["lr"],
            momentum=self.config[model_type]["momentum"],
            weight_decay=float(self.config[model_type]["weight_decay"]),
        )

    def _d_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        real_loss = torch.log(self.model.discriminator(x) + eps)
        fake_loss = torch.log(
            1 - self.model.discriminator(self.model.generator(z)) + eps
        )
        return -torch.mean(real_loss + fake_loss)

    def _g_loss(self, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        return -torch.mean(
            torch.log(self.model.discriminator(self.model.generator(z)) + eps)
        )

    def _log_progress(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        loss_type: str,
    ) -> None:
        print(
            f"[Epoch {epoch:03d}] Step [{step:04d}/{total_steps:04d}] | {loss_type} Loss: {loss:.6f}"
        )

    def _save_images(self, z: torch.Tensor, batches_done: int) -> None:
        gen_imgs = self.model.generator(z).detach().view(-1, 1, 28, 28)
        os.makedirs(self.config["trainer"]["image_save_path"], exist_ok=True)
        save_path = os.path.join(
            self.config["trainer"]["image_save_path"], f"{batches_done}.png"
        )
        save_image(
            gen_imgs, save_path, nrow=int(self.batch_size**0.5), normalize=True
        )

    def train_discriminator(self, epoch: int) -> None:
        total_loss = 0.0
        for i, (x, _) in enumerate(self.train_loader):
            self.optimizer_d.zero_grad()

            x = x.view(x.size(0), -1).to(self.device)
            z = torch.randn(
                x.size(0), self.config["generator"]["hidden_dim"]
            ).to(self.device)

            loss = self._d_loss(x, z)
            total_loss += loss.item()

            loss.backward()
            self.optimizer_d.step()

            self._log_progress(
                epoch,
                i + 1,
                len(self.train_loader),
                loss.item(),
                "Discriminator",
            )

        print(
            f"Average Discriminator Loss: {total_loss / len(self.train_loader)}"
        )

    def train_generator(self, epoch: int) -> None:
        total_loss = 0.0
        for i, _ in enumerate(self.train_loader):
            self.optimizer_g.zero_grad()

            z = torch.randn(
                self.batch_size, self.config["generator"]["hidden_dim"]
            ).to(self.device)
            loss = self._g_loss(z)
            total_loss += loss.item()

            loss.backward()
            self.optimizer_g.step()

            self._log_progress(
                epoch, i + 1, len(self.train_loader), loss.item(), "Generator"
            )

            batches_done = epoch * len(self.train_loader) + i
            if batches_done % self.config["trainer"]["sample_interval"] == 0:
                self._save_images(z, batches_done)

        print(f"Average Generator Loss: {total_loss / len(self.train_loader)}")

    def train(self) -> None:
        torch.manual_seed(self.config["trainer"]["seed"])
        total_epochs = self.config["trainer"]["epochs"]

        for epoch in range(1, total_epochs + 1):
            print(f"\nStarting Epoch [{epoch:03d}/{total_epochs:03d}]")
            print("=" * 50, "\nUpdating Discriminator")
            self.train_discriminator(epoch)

            print("=" * 50, "\nUpdating Generator")
            self.train_generator(epoch)

            print(
                f"[Epoch {epoch:03d}/{total_epochs:03d}] Completed\n{'=' * 50}"
            )
