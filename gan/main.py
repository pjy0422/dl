import yaml
from custom_dataloader import CustomDatasetLoader
from gan import GAN, GANTrainer


def main():
    config = yaml.safe_load(open("config.yaml", "r"))
    dataloader = CustomDatasetLoader(config=config)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader
    model = GAN(config=config)
    print(model.generator)
    print(model.discriminator)
    trainer = GANTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
