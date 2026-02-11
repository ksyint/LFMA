import torch
import torch.nn as nn
from .lfma import LFMAAdapter
from .fourier_ft import FourierFTAdapter
from .lora import LoRALinear



def apply_lfma_to_model(model, target_modules, alpha=12.0, top_k_ratio=0.05):
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                delta_W_init = module.weight.data.clone() * 0.01
                adapter = LFMAAdapter(
                    alpha=alpha,
                    base_layer=module,
                    delta_W_init=delta_W_init,
                    top_k_ratio=top_k_ratio
                )
                setattr(parent, child_name, adapter)
                replaced += 1
    return replaced


def apply_fourier_ft_to_model(model, target_modules, n_freq=1000, alpha=300.0):
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                adapter = FourierFTAdapter(
                    n_freq=n_freq,
                    alpha=alpha,
                    base_layer=module,
                    d1=module.weight.size(0),
                    d2=module.weight.size(1)
                )
                setattr(parent, child_name, adapter)
                replaced += 1
    return replaced


def apply_lora_to_model(model, target_modules, rank=8, alpha=16.0, dropout=0.0):
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                adapter = LoRALinear(
                    base_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                setattr(parent, child_name, adapter)
                replaced += 1
    return replaced


def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def count_adapter_params(model, adapter_cls=None):
    total = 0
    for module in model.modules():
        if adapter_cls is not None and isinstance(module, adapter_cls):
            for p in module.parameters():
                if p.requires_grad:
                    total += p.numel()
        elif adapter_cls is None:
            for p in module.parameters():
                if p.requires_grad:
                    total += p.numel()
    return total


def get_adapter_target_modules(model_type):
    targets = {
        'vit': ['query', 'value'],
        'deit': ['query', 'value'],
        'swin': ['query', 'value'],
        'convnext': ['pwconv1', 'pwconv2'],
        'resnet': ['fc'],
        'clip_vision': ['q_proj', 'v_proj'],
        'clip_text': ['q_proj', 'v_proj'],
        'blip': ['query', 'value'],
        'roberta': ['query', 'value'],
        'gpt2': ['q_proj', 'v_proj'],
        'opt': ['q_proj', 'v_proj'],
        'llama': ['q_proj', 'v_proj'],
    }
    return targets.get(model_type, ['query', 'value'])
