from torch import nn
from network.backbone import get_backbone
from network.classifier import get_classifier
    

class BaseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = 'cpu'
        self._setup(config)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.classifier(features)
        outputs.update({'features': features})
        return outputs

    def _setup(self, config):
        self.backbone = get_backbone(config)
        config.classifier_in_features = self.backbone.out_dim
        self.classifier = get_classifier(config)
    
    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'trainable_params': trainable_params,
            'total_params': total_params
        }
