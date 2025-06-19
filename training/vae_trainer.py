import torch
from torch import nn, optim
import torch.nn.functional as F
from .base import BaseTrainer
from network.base import BaseNet
import logging
from utils.base import Config, weight_init, freeze_model, count_parameters
import numpy as np
from torch.utils.data import DataLoader
import copy
import os


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.generative_feature_dim + cfg.generative_num_classes
        hid    = cfg.generative_hidden_dim
        z_dim  = cfg.generative_latent_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(True),
            nn.Linear(hid, hid),    nn.ReLU(True),
        )
        self.fc_mu     = nn.Linear(hid, z_dim)
        self.fc_logvar = nn.Linear(hid, z_dim)

    def forward(self, x, y_onehot):
        h   = self.net(torch.cat([x, y_onehot], dim=1))
        mu  = self.fc_mu(h)
        log = F.softplus(self.fc_logvar(h)) + 1e-6
        return mu, log


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.generative_latent_dim + cfg.generative_num_classes
        hid    = cfg.generative_hidden_dim
        out_dim= cfg.generative_feature_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(True),
            nn.Linear(hid, hid),    nn.ReLU(True),
            nn.Linear(hid, hid),    nn.ReLU(True),
            nn.Linear(hid, out_dim)                 # no activation
        )

    def forward(self, z, y_onehot):
        return self.net(torch.cat([z, y_onehot], dim=1))


class ConditionalVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)
        self.z_dim = cfg.generative_latent_dim

    def reparameterise(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, y_onehot):
        mu, logvar = self.enc(x, y_onehot)
        z          = self.reparameterise(mu, logvar)
        recon      = self.dec(z, y_onehot)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_div     = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div


def trim(tensor, topk=100):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * topk / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))
    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)

    return trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor)

def merge_task_vectors(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = (gamma_tvs == gamma)
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)

    return mean_tvs

def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in base_params:
        base_tv = base_params[name].clone()
        task_vectors = [task_params[name] for task_params in tasks_params]
        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]

        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        
        params[name] = base_tv + lamb * merged_tv
    
    return params
 

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self._patience = patience
        self._delta = delta
        self._best_score = None
        self._counter = 0
        self._stop = False
        
    def step(self, score):
        if self._best_score is None:
            self._best_score = score
        else:
            if score < self._best_score - self._delta:
                self._best_score = score
                self._counter = 0
            else:
                self._counter += 1
                if self._counter >= self._patience:
                    self._stop = True


class VaeTrainer(BaseTrainer):
    def backbone_path(self, task=-1):
        return "checkpoints/backbone" + (".pth" if task == -1 else f"_{task}.pth")
    
    def cls_path(self, task=-1):
        return "checkpoints/cls" + (".pth" if task == -1 else f"_{task}.pth")

    def pre_train(self):
        torch.save(self.net.get_backbone_trainable_params(), self.backbone_path(-1))
        self.net.to(self.device)
    
    # def setup_task(self):
    #     train_dataset = self.data_manager.get_dataset(np.arange(self.known_classes, self.total_classes), "train", "train")
    #     # self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        
    #     val_frac = 0.1
    #     n_total = len(train_dataset)
    #     n_val = int(n_total * val_frac)
    #     n_train = n_total - n_val
        
    #     train_subset, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
    #     self.train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    #     self.val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

    #     test_dataset = self.data_manager.get_dataset(np.arange(0, self.total_classes), "test", "test")
    #     self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    def merge(self):
        base_params = torch.load(self.backbone_path(-1))
        task_params = [torch.load(self.backbone_path(task)) for task in range(self.cur_task + 1)]
        backbone_params = merge(base_params, task_params, method="ties", lamb=1.0, topk=100)
        self.net.backbone.load_state_dict(backbone_params, strict=False)

    def train(self):
        num_classes = self.data_manager.get_task_size(self.cur_task)
        epochs = self.config.training_epochs
        generative_latent_dim = 256
        total_num_classes = self.data_manager.get_num_classes()

        self.net.eval()

        if self.cur_task == 0:
            generative_config = {
                "generative_latent_dim": generative_latent_dim,
                "generative_hidden_dim": 1024,
                "generative_feature_dim": self.net.backbone.out_dim,
                "generative_num_classes": total_num_classes,
            }
            config = Config(**generative_config)
            vae = ConditionalVAE(config)
            self.vae = vae.to(self.device)
            self.vae.apply(weight_init)
        else:
            vae = self.vae
            vae_old = copy.deepcopy(vae)
            freeze_model(vae_old)

        # Step 1: Optimize the backbone and classifier on current task dataset
        if not os.path.exists(self.backbone_path(self.cur_task)) or not os.path.exists(self.cls_path(self.cur_task)):
            task_net = BaseNet(self.config)
            task_net.classifier.update(num_classes)
            task_net.backbone.load_state_dict(self.net.backbone.state_dict())
            task_net.to(self.device)
            task_net.train()
            task_net_parameters = task_net.count_parameters()
            logging.info(
                f"Task model trainable parameters: {task_net_parameters['trainable_params']}, Total parameters: {task_net_parameters['total_params']}"
            )

            optimizer = optim.SGD(task_net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
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

            task_net.cpu()
            torch.save(
                task_net.get_backbone_trainable_params(), self.backbone_path(self.cur_task)
            )
            torch.save(
                task_net.classifier.classifiers[-1].state_dict(), self.cls_path(self.cur_task)
            )
            del task_net

        self.merge()

        # Step 2: Optimize the generative model
        self.net.eval()

        early_stop = EarlyStopping(patience=10, delta=0.001)
        train_generative_epochs = 200
        total_steps = train_generative_epochs * len(self.train_loader)
        global_step = 0
        optimizer_V = torch.optim.AdamW(vae.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_V, T_max=25, eta_min=1e-5)
        
        for epoch in range(train_generative_epochs):
            vae.train()
            log_loss = {
                "recon": 0.0,
                "kl": 0.0,
                "aug": 0.0,
                "loss": 0.0
            }
            for i, batch in enumerate(self.train_loader):
                global_step += 1
                beta = min(1.0, global_step / (0.1*total_steps))
                _, samples, targets = batch
                samples, targets = samples.to(self.device), targets.to(self.device)

                with torch.no_grad():
                    real_feat = self.net.backbone(samples)
                    real_feat = F.layer_norm(real_feat, real_feat.shape[1:])
                
                y_onehot = F.one_hot(targets, num_classes=total_num_classes).float()
                recon, mu, logvar = vae(real_feat, y_onehot)
                loss, rec_l, kl_l = vae_loss(recon, real_feat, mu, logvar, beta=beta)

                if self.cur_task == 0:
                    loss_aug = torch.tensor(0.0, device=self.device)
                else:
                    aug_B = 256
                    prev_z = torch.randn(aug_B, generative_latent_dim, device=self.device)
                    prev_y = torch.randint(0, self.known_classes, (aug_B,), device=self.device)
                    prev_y_onehot = F.one_hot(prev_y, num_classes=total_num_classes).float()

                    with torch.no_grad():
                        pre_feat = vae_old.dec(prev_z, prev_y_onehot)
                    cur_feat = vae.dec(prev_z, prev_y_onehot)
                    loss_aug = F.mse_loss(cur_feat, pre_feat)

                loss += 5.0 * loss_aug

                torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
                
                optimizer_V.zero_grad()
                loss.backward()
                optimizer_V.step()

                log_loss["recon"] += rec_l.item()
                log_loss["kl"] += kl_l.item()
                log_loss["aug"] += loss_aug.item()
                log_loss["loss"] += loss.item()
            
            scheduler.step()
            
            for key in log_loss:
                log_loss[key] /= len(self.train_loader)
            
            logging.info(
                f"Epoch [{epoch + 1}/{train_generative_epochs}], "
                f"Recon Loss: {log_loss['recon']:.4f}, "
                f"KL Loss: {log_loss['kl']:.4f}, "
                f"Aug Loss: {log_loss['aug']:.4f}, "
                f"Total Loss: {log_loss['loss']:.4f}"
            )
            
            # # validation step
            # vae.eval()
            # val_loss_sum, n = 0.0, 0
            # with torch.no_grad():
            #     for _, samples, targets in self.val_loader:
            #         samples, targets = samples.to(self.device), targets.to(self.device)
                
            #     with torch.no_grad():
            #         real_feat = self.net.backbone(samples)
            #         real_feat = F.layer_norm(real_feat, real_feat.shape[1:])
                
            #     y_onehot = F.one_hot(targets, num_classes=total_num_classes).float()
            #     recon, mu, logvar = vae(real_feat, y_onehot)
            #     elbo, rec_l, kl_l = vae_loss(recon, real_feat, mu, logvar, beta=beta)
                
            #     val_loss_sum += elbo.item() * samples.size(0)
            #     n += samples.size(0)
            
            # val_loss = val_loss_sum / n
            # logging.info(f"Validation Loss: {val_loss:.4f}")
            # early_stop.step(val_loss)
            # if early_stop._stop:
            #     logging.info(f"Early stopping at epoch {epoch + 1} with validation loss {val_loss:.4f}")
            #     break

        # Step 3: Optimize the classifiers with the generated features
        vae.eval()
        self.net.train()

        self.net.classifier.update(num_classes, freeze_old=False)
        self.net.classifier.classifiers[-1].load_state_dict(torch.load(self.cls_path(self.cur_task)), strict=True)
        self.net.classifier.to(self.device)

        classifier_params = {
            "trainable_params": count_parameters(self.net.classifier, trainble=True),
            "total_params": count_parameters(self.net.classifier, trainble=False)
        }
        logging.info(
            f"Classifier trainable parameters: {classifier_params['trainable_params']}, Total parameters: {classifier_params['total_params']}"
        )
        optimizer_C = optim.SGD(self.net.classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs, eta_min=0.0)

        generative_classifier_epochs = 20
        for epoch in range(generative_classifier_epochs):
            total_loss = 0.0
            step = 1000
            B = 128

            for i in range(step):
                with torch.no_grad():
                    labels = torch.randint(0, self.total_classes, (B,), device=self.device)
                    z = torch.randn(B, generative_latent_dim, device=self.device)
                    y_onehot = F.one_hot(labels, total_num_classes).float()
                    synth_feat = vae.dec(z, y_onehot)

                logits = self.net.classifier(synth_feat)['logits']
                loss   = F.cross_entropy(logits, labels)

                optimizer_C.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_C.step()

                total_loss += loss.item()
            
            scheduler_C.step()

            avg_loss = total_loss / step
            logging.info(f"Epoch [{epoch + 1}/{generative_classifier_epochs}], Classifier Loss: {avg_loss:.4f}")

            