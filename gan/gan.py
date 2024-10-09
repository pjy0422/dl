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
        print(f"[Epoch {epoch:03d}/{total_epochs:03d}] Completed\n" + "=" * 50)

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


def gaussian_kernel(x: torch.Tensor, x_sample: torch.Tensor, h: float, d: int):
    x = x.unsqueeze(1)  # Add dimension for broadcasting
    x_sample = x_sample.unsqueeze(0)  # Add dimension for broadcasting
    diffs = x - x_sample  # Pairwise differences between x and x_sample
    distances = torch.sum(diffs**2, dim=-1)  # Squared Euclidean distances

    # Gaussian kernel
    return torch.exp(-distances / (2 * h**2))  # No need for division by d here


def parzen_window_log_likelihood(
    x: torch.Tensor, x_sample: torch.Tensor, h: float, d: int
):
    eps = 1e-12  # To prevent log(0)

    # Compute the Gaussian kernel for all pairs of x and x_sample
    kernel_values = gaussian_kernel(x, x_sample, h, d)

    # Average the kernel values across the generated samples for each real sample
    density_estimates = torch.mean(kernel_values, dim=1)

    # Return the log-likelihood (adding epsilon to avoid log(0))
    return torch.log(density_estimates + eps)


def parzen_window_log_likelihood(
    x: torch.Tensor, x_sample: torch.Tensor, h: float, d: int
):
    eps = 1e-12  # Small value to prevent log(0)

    # Get the kernel values (probability density estimates)
    kernel_values = gaussian_kernel(x, x_sample, h, d)

    # Estimate the density as the mean of the kernel values over the generated samples
    density_estimates = torch.mean(
        kernel_values, dim=1
    )  # Average over the second dimension (samples)

    # Compute the log-likelihood
    return torch.log(density_estimates + eps)


def main():
    config = {
        "dataset": {"width": 28, "height": 28, "channels": 1},
        "generator": {"hidden_dim": 256},
        "discriminator": {
            "latent_dim": 256,
            "dropout": 0.2,
            "num_pieces": 5,
        },
        "trainer": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 64,
            "epochs": 10,
            "lr": 0.0002,
            "momentum": 0.5,
            "seed": 42,
        },
    }

    model = Generator(hidden_dim=256, output_dim=28 * 28)

    input_data = torch.randn(64, 1, 28, 28)
    latent_data = torch.randn(64, 256)
    gen_output = model(latent_data)

    x = input_data.view(input_data.size(0), -1).unsqueeze(0)
    x_sample = gen_output.unsqueeze(1)

    diffs = x - x_sample
    print((x - x_sample).shape)
    distances = torch.sum(diffs**2, dim=-1)
    print(distances.shape)
    print(parzen_window_log_likelihood(x, x_sample, 6, 784))


if __name__ == "__main__":
    main()
