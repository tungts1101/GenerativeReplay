import torch
from torch import nn, optim
import torch.nn.functional as F
from .base import BaseTrainer
from network.base import BaseNet
import logging
from utils.base import Config, weight_init, freeze_model, count_parameters
from torch.nn.utils import spectral_norm
import copy
import os


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.generative_latent_dim + config.generative_num_classes # conditional generator
        self.hidden_dim = config.generative_hidden_dim
        self.feature_dim = config.generative_feature_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.apply(weight_init)

    def forward(self, z, y):
        """
        z: latent vector of shape (batch_size, latent_dim)
        y: one-hot encoded class labels of shape (batch_size, num_classes)
        """
        z_y = torch.cat((z, y), dim=1)
        f = self.model(z_y)
        return f


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.generative_feature_dim
        self.hidden_dim = config.discriminator_hidden_dim
        self.total_classes = config.generative_num_classes

        self.model = nn.Sequential(
            spectral_norm(nn.Linear(self.feature_dim, self.hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.valid = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.total_classes)
        self.projection = nn.Linear(self.total_classes, self.hidden_dim, bias=False)
        self.apply(weight_init)
    
    def forward(self, z, y):
        h = self.model(z)
        var = self.projection(y)
        validity = (var * h).sum(dim=1).reshape(-1, 1) + self.valid(h)
        logits = self.classifier(h)
        return validity, logits


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


def compute_gradient_penalty(D, real_feat, fake_feat, y_onehot):
    """Return per-sample GP (no mean yet)"""
    bs = real_feat.size(0)
    a  = torch.rand(bs, 1, device=real_feat.device)
    a  = a.expand_as(real_feat)

    inter = (a * real_feat + (1 - a) * fake_feat).requires_grad_(True)
    out, _ = D(inter, y_onehot)

    grad = torch.autograd.grad(
        outputs   = out,
        inputs    = inter,
        grad_outputs = torch.ones_like(out),
        create_graph = True,
        retain_graph = True,
        only_inputs  = True
    )[0]

    grad = grad.view(bs, -1)
    gp   = (grad.norm(2, dim=1) - 1) ** 2         # shape = (bs,)
    return gp  


class FeatureGeneratorTrainer(BaseTrainer):
    def backbone_path(self, task=-1):
        return "checkpoints/backbone" + (".pth" if task == -1 else f"_{task}.pth")
    
    def cls_path(self, task=-1):
        return "checkpoints/cls" + (".pth" if task == -1 else f"_{task}.pth")

    def pre_train(self):
        torch.save(self.net.get_backbone_trainable_params(), self.backbone_path(-1))
        self.net.to(self.device)
    
    def merge(self):
        base_params = torch.load(self.backbone_path(-1))
        task_params = [torch.load(self.backbone_path(task)) for task in range(self.cur_task + 1)]
        backbone_params = merge(base_params, task_params, method="ties", lamb=1.0, topk=100)
        self.net.backbone.load_state_dict(backbone_params, strict=False)

    def train(self):
        num_classes = self.data_manager.get_task_size(self.cur_task)
        epochs = self.config.training_epochs
        generative_latent_dim = 200
        total_num_classes = self.data_manager.get_num_classes()

        self.net.eval()

        if self.cur_task == 0:
            generative_config = {
                "generative_latent_dim": generative_latent_dim,
                "generative_hidden_dim": 1024,
                "discriminator_hidden_dim": 512,
                "generative_feature_dim": self.net.backbone.out_dim,
                "generative_num_classes": total_num_classes,
            }
            config = Config(**generative_config)

            generator = Generator(config)
            discriminator = Discriminator(config)
            generator.to(self.device)
            discriminator.to(self.device)
            self.generator = generator
            self.discriminator = discriminator
        else:
            generator = self.generator
            discriminator = self.discriminator
            generator_old = copy.deepcopy(generator)
            generator_old.eval()
            freeze_model(generator_old)

        generator.eval()
        discriminator.eval()

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
                total_acc = 0.0
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
                    preds = outputs.argmax(dim=1)
                    total_acc += (preds == targets).sum().item()

                scheduler.step()

                avg_loss = total_loss / len(self.train_loader)
                avg_acc = total_acc / len(self.train_loader.dataset)
                logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

            task_net.cpu()
            torch.save(
                task_net.get_backbone_trainable_params(), self.backbone_path(self.cur_task)
            )
            torch.save(
                task_net.classifier.classifiers[-1].state_dict(), self.cls_path(self.cur_task)
            )
            del task_net

        self.merge()
        self.net.eval()

        # Step 2: Optimize the generator and discriminator
        if (os.path.exists(f"checkpoints/generator_{self.cur_task}.pth") and os.path.exists(f"checkpoints/discriminator_{self.cur_task}.pth")) and False:
            generator.load_state_dict(torch.load(f"checkpoints/generator_{self.cur_task}.pth"))
            discriminator.load_state_dict(torch.load(f"checkpoints/discriminator_{self.cur_task}.pth"))
        else:
            train_generative_epochs = 50
            num_critic = 3
            lambda_gp = 10.0
            lambda_d_cls = 5.0
            lambda_g_cls = 1.0
            lambda_g_aug = 10.0

            optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.0, 0.9))
            scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=train_generative_epochs, eta_min=0.0)
            scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=train_generative_epochs, eta_min=0.0)

            for epoch in range(train_generative_epochs):
                loss_log = {
                    'D/loss': 0.0,
                    'D/new_rf': 0.0,
                    'D/new_cls': 0.0,
                    'D/new_gp': 0.0,
                    'G/loss': 0.0,
                    'G/new_rf': 0.0,
                    'G/new_cls': 0.0,
                    'G/new_aug': 0.0,
                }

                for i, batch in enumerate(self.train_loader):
                    _, samples, targets = batch
                    B = samples.size(0)
                    samples, targets = samples.to(self.device), targets.to(self.device)

                    # train discriminator
                    generator.eval()
                    discriminator.train()

                    for _ in range(num_critic):
                        z = torch.randn(B, generative_latent_dim, device=self.device)
                        y_onehot = F.one_hot(targets, num_classes=total_num_classes).float()

                        with torch.no_grad():
                            real_feat = self.net.backbone(samples)
                            fake_feat = generator(z, y_onehot)

                        fake_validity, _ = discriminator(fake_feat, y_onehot)
                        real_validity, disc_real_acgan = discriminator(real_feat, y_onehot)

                        d_loss_rf = -torch.mean(real_validity) + torch.mean(fake_validity)
                        d_loss_cls = F.cross_entropy(disc_real_acgan, targets)
                        gradient_penalty = compute_gradient_penalty(discriminator, real_feat, fake_feat, y_onehot).mean()
                        d_loss = d_loss_rf + lambda_gp * gradient_penalty + lambda_d_cls * d_loss_cls
                        # d_loss = d_loss_rf + lambda_d_cls * d_loss_cls

                        optimizer_D.zero_grad()
                        d_loss.backward()
                        optimizer_D.step()

                        loss_log['D/loss'] += d_loss.item()
                        loss_log['D/new_rf'] += d_loss_rf.item()
                        loss_log['D/new_cls'] += d_loss_cls.item() if lambda_d_cls != 0 else 0
                        loss_log['D/new_gp'] += gradient_penalty.item() if lambda_gp != 0 else 0

                    # train generator
                    discriminator.eval()
                    generator.train()

                    z = torch.randn(B, generative_latent_dim, device=self.device)
                    y_onehot = F.one_hot(targets, num_classes=total_num_classes).float()

                    fake_feat = generator(z, y_onehot)
                    fake_validity, disc_fake_acgan = discriminator(fake_feat, y_onehot)

                    g_loss_rf = -torch.mean(fake_validity)
                    g_loss_cls = F.cross_entropy(disc_fake_acgan, targets)

                    if self.cur_task == 0:
                        loss_aug = torch.tensor(0.0, device=self.device)
                    else:
                        aug_B = 256
                        prev_z = torch.randn(aug_B, generative_latent_dim, device=self.device)
                        prev_y = torch.randint(0, self.known_classes, (aug_B,), device=self.device)
                        prev_y_onehot = F.one_hot(prev_y, num_classes=total_num_classes).float()

                        with torch.no_grad():
                            pre_feat = generator_old(prev_z, prev_y_onehot)
                        cur_feat = generator(prev_z, prev_y_onehot)
                        loss_aug = F.mse_loss(cur_feat, pre_feat)

                    g_loss = g_loss_rf + lambda_g_cls * g_loss_cls + lambda_g_aug * loss_aug

                    loss_log['G/loss'] += g_loss.item()
                    loss_log['G/new_rf'] += g_loss_rf.item()
                    loss_log['G/new_cls'] += g_loss_cls.item() if lambda_g_cls != 0 else 0
                    loss_log['G/new_aug'] = loss_aug.item() if lambda_g_aug != 0 else 0

                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                    
                # scheduler_D.step()
                # scheduler_G.step()
                for key in loss_log:
                    loss_log[key] /= len(self.train_loader)
                logging.info(f"Epoch [{epoch + 1}/{train_generative_epochs}], Loss Log: {loss_log}")
            
            torch.save(generator.state_dict(), f"checkpoints/generator_{self.cur_task}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_{self.cur_task}.pth")

        # Step 3: Optimize the classifiers with the generated features
        generator.eval()
        discriminator.eval()
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
                labels      = torch.randint(0, self.total_classes, (B,), device=self.device)
                y_onehot    = F.one_hot(labels, total_num_classes).float()
                z           = torch.randn(B, generative_latent_dim, device=self.device)

                with torch.no_grad():
                    synth_feat = generator(z, y_onehot)

                logits = self.net.classifier(synth_feat)['logits']
                loss   = F.cross_entropy(logits, labels)

                optimizer_C.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_C.step()

                total_loss += loss.item()
            
            scheduler_C.step()

            avg_loss = total_loss / step
            logging.info(f"Epoch [{epoch + 1}/{generative_classifier_epochs}], Classifier Loss: {avg_loss:.4f}")

            