import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA


class CustomDatasetLoader:
    def __init__(
        self,
        config: str,
        transform: transforms.Compose = None,
    ) -> None:

        with open(config, "r") as file:
            config = yaml.safe_load(file)
        dataset = config["dataset"]
        self.dataset_name = dataset["name"]
        self.batch_size = dataset["batch_size"]
        self.train = dataset["train"]
        self.val_split = dataset["val_split"]
        self.download = dataset["download"]
        self.num_workers = dataset["num_workers"]
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
            return transforms.Compose([transforms.ToTensor()])
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
