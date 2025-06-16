import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from dataset.data import DataManager
from network.base import BaseNet
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config
        self.data_manager: DataManager = config.data_manager
        self.net: BaseNet = config.net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.device = self.device

        self.cur_task = -1
        self.known_classes = 0
        self.total_classes = 0
        self.class_increments = []

    def run(self):
        self._set_random()
        self.pre_train()

        num_tasks = self.data_manager.get_num_tasks()
        for task in range(num_tasks):
            self.pre_task()
            self.setup_task()

            logging.info(
                f"Training on task {task + 1}, classes {self.known_classes} - {self.total_classes-1}"
            )
            self.train()

            logging.info(f"Testing on task {task + 1}")
            self.test()

            self.post_task()

    def pre_train(self):
        self.net.to(self.device)

    def pre_task(self):
        self.cur_task += 1
        self.total_classes += self.data_manager.get_task_size(self.cur_task)
        self.class_increments.append((self.known_classes, self.total_classes - 1))

    def post_task(self):
        self.known_classes = self.total_classes
    
    def setup_task(self):
        train_dataset = self.data_manager.get_dataset(np.arange(self.known_classes, self.total_classes), "train", "train")
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

        test_dataset = self.data_manager.get_dataset(np.arange(0, self.total_classes), "test", "test")
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    @abstractmethod
    def train(self):
        pass

    def test(self):
        self.net.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                _, samples, targets = batch
                samples, targets = samples.to(self.device), targets.to(self.device)
                outputs = self.net(samples)['logits']
                predicts = outputs.argmax(dim=1)
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        acc_total, grouped = self._accuracy(y_pred, y_true, self.class_increments)
        logging.info(f"Total accuracy: {acc_total}")
        logging.info(f"Grouped accuracy: {grouped}")

    def _accuracy(self, y_pred, y_true, class_increments):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        acc_total = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )

        for classes in class_increments:
            idxes = np.where(
                np.logical_and(y_true >= classes[0], y_true <= classes[1])
            )[0]
            label = "{}-{}".format(
                str(classes[0]).rjust(2, "0"), str(classes[1]).rjust(2, "0")
            )
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )

        return acc_total, all_acc

    def _set_random(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
