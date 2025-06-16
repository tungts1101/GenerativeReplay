from dataset.data import DataManager
from network.base import BaseNet
from training.simple_trainer import SimpleTrainer
from training.feature_generator_trainer import FeatureGeneratorTrainer
import logging
import sys
from utils.base import Config


parsed_config = {
    "seed": 42,

    "dataset_root": "./data",
    "dataset_name": "cifar224",
    "dataset_shuffle": True,
    "dataset_increments": [10 for _ in range(10)],

    "net_backbone_name": "vit_base_patch16_224_lora",

    "training_epochs": 1,
}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    config = Config(**parsed_config)
    data_manager = DataManager(config)
    config.data_manager = data_manager
    config.net = BaseNet(config)

    trainer = FeatureGeneratorTrainer(config)
    trainer.run()
