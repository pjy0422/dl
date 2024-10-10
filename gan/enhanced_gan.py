import os

import mlflow
import torch
import torch.nn as nn
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
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

        # Initialize learning rate schedulers
        self.scheduler_g = self._init_scheduler(self.optimizer_g, "generator")
        self.scheduler_d = self._init_scheduler(
            self.optimizer_d, "discriminator"
        )

    def _init_optimizer(self, model, model_type: str) -> torch.optim.Optimizer:
        b1, b2 = (float(item) for item in self.config[model_type]["betas"])
        return torch.optim.Adam(
            model.parameters(),
            lr=float(self.config[model_type]["lr"]),
            betas=(b1, b2),
        )

    def _init_scheduler(
        self, optimizer, model_type: str
    ) -> torch.optim.lr_scheduler._LRScheduler:
        # Using StepLR scheduler here; you can modify this based on your requirement
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(
                self.config[model_type]["lr_scheduler"]["step_size"]
            ),
            gamma=float(self.config[model_type]["lr_scheduler"]["gamma"]),
        )

    def _d_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        real = torch.ones(x.size(0), 1).to(self.device)
        fake = torch.zeros(x.size(0), 1).to(self.device)
        loss_fn = nn.BCELoss()
        real_loss = loss_fn(self.model.discriminator(x), real)
        fake_loss = loss_fn(
            self.model.discriminator(self.model.generator(z)), fake
        )
        return torch.mean(real_loss + fake_loss)

    def _g_loss(self, z: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.BCELoss()
        valid = torch.ones(z.size(0), 1).to(self.device)
        return loss_fn(
            self.model.discriminator(self.model.generator(z)), valid
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
        # Log the loss to MLflow
        mlflow.log_metric(
            f"{loss_type}_loss", loss, step=epoch * total_steps + step
        )

    def _save_images(self, z: torch.Tensor, batches_done: int) -> None:
        gen_imgs = (
            self.model.generator(z)
            .detach()
            .view(
                -1,
                1,
                self.config["dataset"]["height"],
                self.config["dataset"]["width"],
            )
        )
        os.makedirs(self.config["trainer"]["image_save_path"], exist_ok=True)
        save_path = os.path.join(
            self.config["trainer"]["image_save_path"], f"{batches_done}.png"
        )
        save_image(
            gen_imgs, save_path, nrow=int(self.batch_size**0.5), normalize=True
        )
        # Log the generated images to MLflow
        mlflow.log_artifact(save_path)

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

        avg_loss = total_loss / len(self.train_loader)
        print(f"Average Discriminator Loss: {avg_loss}")
        mlflow.log_metric("avg_discriminator_loss", avg_loss, epoch)

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

        avg_loss = total_loss / len(self.train_loader)
        print(f"Average Generator Loss: {avg_loss}")
        mlflow.log_metric("avg_generator_loss", avg_loss, epoch)

    def log_learning_rates(self, epoch: int) -> None:
        # Log the learning rate for both generator and discriminator
        for param_group in self.optimizer_g.param_groups:
            lr_g = param_group["lr"]
        for param_group in self.optimizer_d.param_groups:
            lr_d = param_group["lr"]

        mlflow.log_metric("learning_rate_generator", lr_g, epoch)
        mlflow.log_metric("learning_rate_discriminator", lr_d, epoch)

    def train(self) -> None:
        torch.manual_seed(self.config["trainer"]["seed"])
        total_epochs = self.config["trainer"]["epochs"]

        for epoch in range(1, total_epochs + 1):
            print(f"\nStarting Epoch [{epoch:03d}/{total_epochs:03d}]")
            print("=" * 50, "\nUpdating Discriminator")
            self.train_discriminator(epoch)

            print("=" * 50, "\nUpdating Generator")
            self.train_generator(epoch)

            # Log learning rates at the end of each epoch
            self.log_learning_rates(epoch)

            # Step the learning rate schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()

            print(
                f"[Epoch {epoch:03d}/{total_epochs:03d}] Completed\n{'=' * 50}"
            )
