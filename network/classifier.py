import torch
from torch import nn
from timm.models.layers.weight_init import trunc_normal_


def get_classifier(config):
    classifier_name = getattr(config, 'net_classifier_name', 'inc_linear')
    if classifier_name == 'inc_linear':
        return IncrementalLinearClassifier(config)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
    

class IncrementalLinearClassifier(nn.Module):
    def __init__(self, config):
        assert hasattr(config, 'classifier_in_features'), "config must have 'classifier_in_features'"

        super().__init__()
        self.in_features = config.classifier_in_features
        self.use_bias = getattr(config, 'classifier_use_bias', False)
        self.classifiers = nn.ModuleList()
    
    def update(self, num_classes, freeze_old=True):
        if freeze_old:
            self.freeze()
        classifier = nn.Linear(self.in_features, num_classes, bias=self.use_bias)
        trunc_normal_(classifier.weight, std=0.02)
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0)
        self.classifiers.append(classifier)
    
    def freeze(self):
        for classifier in self.classifiers:
            for param in classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = [classifier(x) for classifier in self.classifiers]
        return {'logits': torch.cat(outputs, dim=1)}