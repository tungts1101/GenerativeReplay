from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import logging
from torchvision import transforms
import torch.utils.data as data


class iDataset:
    def __init__(self, config):
        self.config = config

    @property
    def use_path(self):
        return self.use_path

    @property
    def class_order(self):
        return self.class_order

    @property
    def train_transform(self):
        return self.train_transform

    @property
    def test_transform(self):
        return self.test_transform

    @property
    def common_transform(self):
        return self.common_transform

    def get_train_dataset(self):
        return self.train_samples, self.train_targets

    def get_test_dataset(self):
        return self.test_samples, self.test_targets


class DataManager:
    def __init__(self, config):
        self._setup_dataset(config)
        self.increments = config.dataset_increments

        offset = len(self.class_order) - sum(self.increments)
        if offset > 0:
            self.increments.append(offset)

    def get_num_tasks(self):
        return len(self.increments)

    def get_task_size(self, task_index):
        return self.increments[task_index]

    def get_num_classes(self):
        return len(self.class_order)

    def get_dataset(self, indices, source, mode):
        if source == "train":
            samples, targets = self.train_samples, self.train_targets
        elif source == "test":
            samples, targets = self.test_samples, self.test_targets
        else:
            raise ValueError(
                f"Source {source} is not supported. Use 'train' or 'test'."
            )

        if mode == "train":
            transform = transforms.Compose(
                [*self.train_transform, *self.common_transform]
            )
        elif mode == "test":
            transform = transforms.Compose(
                [*self.test_transform, *self.common_transform]
            )
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'train' or 'test'.")

        selected_samples, selected_targets = [], []
        for idx in indices:
            class_samples, class_targets = self._select(
                samples, targets, low=idx, high=idx + 1
            )
            selected_samples.append(class_samples)
            selected_targets.append(class_targets)

        selected_samples = np.concatenate(selected_samples)
        selected_targets = np.concatenate(selected_targets)

        return Dataset(selected_samples, selected_targets, transform, self.use_path)

    def _setup_dataset(self, config):
        dataset_name = config.dataset_name.lower()
        dataset = self._get_dataset(dataset_name, config)
        self.train_samples, self.train_targets = dataset.get_train_dataset()
        self.test_samples, self.test_targets = dataset.get_test_dataset()
        self.use_path = dataset.use_path

        self.train_transform = dataset.train_transform
        self.test_transform = dataset.test_transform
        self.common_transform = dataset.common_transform

        if config.dataset_shuffle:
            order = [i for i in range(len(np.unique(self.train_targets)))]
            np.random.seed(config.seed)
            order = np.random.permutation(order).tolist()
        else:
            order = dataset.class_order
        self.class_order = order
        logging.info(
            f"Train samples: {len(self.train_samples)}, Test samples: {len(self.test_samples)}"
        )
        logging.info(
            f"Dataset {dataset_name} loaded with class order: {self.class_order}"
        )

        self.train_targets = self._map_new_class_index(
            self.train_targets, self.class_order
        )
        self.test_targets = self._map_new_class_index(
            self.test_targets, self.class_order
        )

    def _map_new_class_index(self, targets, class_order):
        return np.array(list(map(lambda x: class_order.index(x), targets)))

    def _select(self, samples, targets, low, high):
        indexes = np.where((targets >= low) & (targets < high))[0]
        return samples[indexes], targets[indexes]

    def _get_dataset(self, dataset_name, config):
        if dataset_name == "cifar10":
            from dataset.icifar import iCIFAR10

            return iCIFAR10(config)
        elif dataset_name == "cifar100":
            from dataset.icifar import iCIFAR100

            return iCIFAR100(config)
        elif dataset_name == "cifar224":
            from dataset.icifar import iCIFAR224

            return iCIFAR224(config)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")


class Dataset(data.Dataset):
    def __init__(self, samples, targets, transform=None, use_path=False):
        self.samples = samples
        self.targets = targets
        self.transform = transform
        self.use_path = use_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.use_path:
            sample = self.transform(pil_loader(self.samples[index]))
        else:
            sample = self.transform(Image.fromarray(self.samples[index]))
        target = self.targets[index]

        return index, sample, target


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
