import timm
from peft import LoraConfig, get_peft_model


def get_backbone(config):
    backbone_name = getattr(config, 'net_backbone_name', 'resnet18')
    backbone_pretrained = getattr(config, 'net_backbone_pretrained', True)

    if backbone_name == 'resnet18':
        backbone = timm.create_model('resnet18', pretrained=backbone_pretrained, num_classes=0)
        backbone.out_dim = 512
        return backbone
    elif backbone_name == "vit_base_patch16_224":
        backbone = timm.create_model('vit_base_patch16_224', pretrained=backbone_pretrained, num_classes=0)
        backbone.out_dim = 768
        return backbone
    elif backbone_name == "vit_base_patch16_224_lora":
        backbone = timm.create_model('vit_base_patch16_224', pretrained=backbone_pretrained, num_classes=0)
        backbone.out_dim = 768
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian"
        )
        return get_peft_model(backbone, lora_config)
    else:
        raise ValueError(f"Unsupported backbone encoder: {backbone_name}")