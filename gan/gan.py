import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA


class CustomDatasetLoader:
    def __init__(
        self,
        dataset_name: str = "mnist",
        batch_size: int = 64,
        train: bool = True,
        val_split: float = 0.1,
        download: bool = True,
        num_workers: int = 4,
        transform: transforms.Compose = None,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.train = train
        self.val_split = val_split
        self.download = download
        self.num_workers = num_workers
        self.transform = transform or self._get_transform()

        self.dataset = self._get_dataset()

        if self.train and self.val_split > 0:
            self.train_data, self.val_data = self._split_dataset()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.train:
            self.train_loader, self.val_loader = self._get_train_val_loaders()
        else:
            self.test_loader = self._get_test_loader()

    def _get_transform(self) -> transforms.Compose:
        if self.dataset_name == "cifar10":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name == "mnist":
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        elif self.dataset_name == "celeba":
            return transforms.Compose(
                [
                    transforms.CenterCrop(178),  # Fixed typo
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            raise ValueError(
                f"Unknown dataset {self.dataset_name}. Supported datasets: 'mnist', 'cifar10', 'celeba'."
            )

    def _get_dataset(self):
        if self.dataset_name == "cifar10":
            return CIFAR10(
                root="./data",
                train=self.train,
                download=self.download,
                transform=self.transform,
            )
        elif self.dataset_name == "mnist":
            return MNIST(
                root="./data",
                train=self.train,
                download=self.download,
                transform=self.transform,
            )
        elif self.dataset_name == "celeba":
            return CelebA(
                root="./data",
                split="train" if self.train else "test",
                download=self.download,
                transform=self.transform,
            )
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}.")

    def _split_dataset(self) -> tuple:
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        return random_split(self.dataset, [train_size, val_size])

    def _get_train_val_loaders(self) -> tuple:
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader

    def _get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 28 * 28,
    ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(2 * self.latent_dim, output_dim),
        )

    def forward(self, z) -> torch.Tensor:
        """
        args:
            z: torch.Tensor - expected shape [batch_size, latent_dim] gaussian noise
        return:
            torch.Tensor - generated image
        """

        return self.model(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> torch.Tensor:
        """
        args:
            x: torch.Tensor - expected shape [batch_size, input_dim] input data
        return:
            torch.Tensor - predicted value
        """

        return self.model(x)


def main():
    mnist = CustomDatasetLoader(
        dataset_name="mnist", batch_size=64, train=True
    )
    train_loader = mnist.train_loader
    val_loader = mnist.val_loader


if __name__ == "__main__":
    main()
