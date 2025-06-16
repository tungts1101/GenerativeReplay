import torch
from torch import nn, optim
import torch.nn.functional as F
from network.base import BaseNet
from .base import BaseTrainer
import logging


class SimpleTrainer(BaseTrainer):
    def train(self):
        num_classes = self.data_manager.get_task_size(self.cur_task)

        task_net = BaseNet(self.config)
        task_net.classifier.update(num_classes)
        task_net.backbone.load_state_dict(self.net.backbone.state_dict())
        task_net.backbone.requires_grad_(False)
        task_net.to(self.device)
        task_net.train()
        task_net_parameters = task_net.count_parameters()
        logging.info(
            f"Task model trainable parameters: {task_net_parameters['trainable_params']}, Total parameters: {task_net_parameters['total_params']}"
        )

        epochs = self.config.training_epochs

        optimizer = optim.SGD(
            task_net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                _, samples, targets = batch
                samples, targets = samples.to(self.device), targets.to(self.device)
                targets = torch.where(targets >= self.known_classes, targets - self.known_classes, -100)

                outputs = task_net(samples)["logits"]
                loss = F.cross_entropy(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        self.net.classifier.update(num_classes)
        self.net.classifier.to(self.device) 
        self.net.backbone.load_state_dict(task_net.backbone.state_dict())
        self.net.classifier.classifiers[-1].load_state_dict(
            task_net.classifier.classifiers[-1].state_dict()
        )
