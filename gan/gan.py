import mlflow
import torch
import torch.nn as nn
from maxout import MaxOut


class Generator(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, output_dim: int = 28 * 28
    ) -> None:
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.model = self._build_model(output_dim)

    def _build_model(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, output_dim),
            nn.Sigmoid(),
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

    def _build_model(
        self, input_dim: int, latent_dim: int, dropout: float, num_pieces: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            MaxOut(input_dim=latent_dim, num_pieces=num_pieces),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
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
        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()

    def _init_generator(self) -> Generator:
        return Generator(
            hidden_dim=self.config["generator"]["hidden_dim"],
            output_dim=self._calc_output_dim(),
        )

    def _init_discriminator(self) -> Discriminator:
        return Discriminator(
            input_dim=self._calc_output_dim(),
            latent_dim=self.config["discriminator"]["latent_dim"],
            dropout=self.config["discriminator"]["dropout"],
            num_pieces=self.config["discriminator"]["num_pieces"],
        )

    def _calc_output_dim(self) -> int:
        return (
            self.config["dataset"]["width"]
            * self.config["dataset"]["height"]
            * self.config["dataset"]["channels"]
        )


class GANTrainer:
    def __init__(
        self,
        model: GAN,
        config: dict,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.device = config["trainer"]["device"]
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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

    def _gaussian_kernel(
        self, x: torch.Tensor, x_sample: torch.Tensor, sigma: float
    ):
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        x_sample = x_sample.unsqueeze(0)  # Shape: (1, num_samples, input_dim)
        diffs = x - x_sample  # Broadcasting subtraction
        distances = torch.sum(diffs**2, dim=-1)  # Squared Euclidean distances
        return torch.exp(-distances / (2 * sigma**2))

    def _parzen_window_log_likelihood(
        self, x: torch.Tensor, x_sample: torch.Tensor, sigma: float
    ):
        eps = 1e-12  # Prevent log(0)
        kernel_values = self._gaussian_kernel(
            x, x_sample, sigma=sigma
        )  # Shape: (batch_size, num_samples)
        density_estimates = torch.mean(
            kernel_values, dim=1
        )  # Mean over generated samples
        log_likelihood = torch.log(density_estimates + eps)
        return torch.mean(log_likelihood)  # Average over validation data

    def _print_progress(
        self,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
        loss_type: str,
    ) -> None:
        print(
            f"[Epoch {epoch:03d}/{total_epochs:03d}] "
            f"Step [{step:04d}/{total_steps:04d}] | {loss_type:<12} Loss: {loss:.6f}"
        )

    def _print_epoch_summary(self, epoch: int, total_epochs: int) -> None:
        print(f"[Epoch {epoch:03d}/{total_epochs:03d}] Completed")
        print("Evaluating Model".center(50, "="))

    def train_discriminator(self, epoch: int, total_epochs: int) -> None:
        for i, (x, _) in enumerate(self.train_dataloader):
            self.optimizer_d.zero_grad()

            x = x.view(x.size(0), -1).to(self.device)
            z = torch.randn(
                x.size(0), self.config["generator"]["hidden_dim"]
            ).to(self.device)

            loss = self._d_loss(x, z)
            loss.backward()
            self.optimizer_d.step()

            self._print_progress(
                epoch,
                total_epochs,
                i + 1,
                len(self.train_dataloader),
                loss.item(),
                "Discriminator",
            )

    def train_generator(self, epoch: int, total_epochs: int) -> None:
        self.optimizer_g.zero_grad()

        z = torch.randn(
            self.batch_size, self.config["generator"]["hidden_dim"]
        ).to(self.device)
        loss = self._g_loss(z)
        loss.backward()
        self.optimizer_g.step()

        self._print_progress(
            epoch, total_epochs, 1, 1, loss.item(), "Generator"
        )

    def train(self) -> None:
        torch.manual_seed(self.config["trainer"]["seed"])

        total_epochs = self.config["trainer"]["epochs"]
        for epoch in range(1, total_epochs + 1):
            print(f"\nStarting Epoch [{epoch:03d}/{total_epochs:03d}]")
            print("Updating Discriminator".center(50, "="))
            self.train_discriminator(epoch, total_epochs)

            print("Updating Generator".center(50, "="))
            self.train_generator(epoch, total_epochs)

            self._print_epoch_summary(epoch, total_epochs)
            # Evaluate after each epoch
            self.eval()

    def eval(self) -> None:
        log_likelihoods = []
        sigma = 0.5  # You might want to tune this

        with torch.no_grad():  # Disable gradients for evaluation
            for i, (x, _) in enumerate(self.val_dataloader):
                x = x.view(x.size(0), -1).to(self.device)
                z = torch.randn(
                    x.size(0), self.config["generator"]["hidden_dim"]
                ).to(self.device)

                log_likelihood = self._parzen_window_log_likelihood(
                    x, self.model.generator(z), sigma=sigma
                )
                log_likelihoods.append(log_likelihood.item())

        mean_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        print(f"Mean Log-Likelihood: {mean_log_likelihood:.6f}")
