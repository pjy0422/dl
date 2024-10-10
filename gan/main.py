import mlflow
import mlflow.pytorch
import torch
import yaml
from custom_dataloader import CustomDatasetLoader
from enhanced_gan import GAN, GANTrainer


def load_models(run_id):
    # Load the generator model
    generator_path = f"runs:/{run_id}/models/generator"
    generator = mlflow.pytorch.load_model(generator_path)

    # Load the discriminator model
    discriminator_path = f"runs:/{run_id}/models/discriminator"
    discriminator = mlflow.pytorch.load_model(discriminator_path)

    return generator, discriminator


def main():

    config = yaml.safe_load(open("config.yaml", "r"))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run() as run:
        mlflow.log_params(config)

        dataloader = CustomDatasetLoader(config=config)
        train_loader = dataloader.train_loader
        val_loader = dataloader.val_loader
        model = GAN(config=config)
        print(model.generator)
        print(model.discriminator)
        trainer = GANTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.train()

        # Create an input example (using a random tensor or data from your loader)
        mlflow.pytorch.log_model(
            trainer.model.generator,
            artifact_path="models/generator",
            conda_env="./environment.yaml",
        )
        mlflow.pytorch.log_model(
            trainer.model.discriminator,
            artifact_path="models/discriminator",
            conda_env="./environment.yaml",
        )
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    mlflow.end_run()
    print("Finished training and logging to MLflow.")
    generator_model, discriminator_model = load_models(run_id)
    print(generator_model)
    print(discriminator_model)


if __name__ == "__main__":
    main()
