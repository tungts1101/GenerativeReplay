from torchvision import datasets, transforms
from .data import iDataset
import numpy as np


class iCIFAR10(iDataset):
    use_path = False
    class_order = np.arange(10).tolist()
    train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_transform = []
    common_transform = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    def __init__(self, config):
        super().__init__(config)
        train_dataset = datasets.CIFAR10(
            root=config.dataset_root, train=True, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=config.dataset_root, train=False, download=True
        )

        self.train_samples, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_samples, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iDataset):
    use_path = False
    class_order = np.arange(100).tolist()
    train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_transform = []
    common_transform = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    def __init__(self, config):
        super().__init__(config)
        train_dataset = datasets.CIFAR100(
            root=config.dataset_root, train=True, download=True
        )
        test_dataset = datasets.CIFAR100(
            root=config.dataset_root, train=False, download=True
        )

        self.train_samples, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_samples, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR224(iDataset):
    use_path = False
    class_order = np.arange(100).tolist()
    train_transform = [
        transforms.RandomResizedCrop(
            size=224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    test_transform = [
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
    ]
    common_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    def __init__(self, config):
        super().__init__(config)
        train_dataset = datasets.CIFAR100(
            root=config.dataset_root, train=True, download=True
        )
        test_dataset = datasets.CIFAR100(
            root=config.dataset_root, train=False, download=True
        )

        self.train_samples, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_samples, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
