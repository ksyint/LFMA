import torch
import torch.nn as nn
from transformers import (

    ViTForImageClassification,
    AutoImageProcessor,
    DeiTForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
    ResNetForImageClassification,
)


VISION_MODELS = {
    'vit_base': 'google/vit-base-patch16-224-in21k',
    'vit_large': 'google/vit-large-patch16-224-in21k',
    'deit_base': 'facebook/deit-base-patch16-224',
    'swin_base': 'microsoft/swin-base-patch4-window7-224',
    'convnext_base': 'facebook/convnext-base-224',
}


def load_vit(model_name, num_labels):
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def load_deit(model_name, num_labels):
    model = DeiTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def load_swin(model_name, num_labels):
    model = SwinForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def load_convnext(model_name, num_labels):
    model = ConvNextForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def load_resnet(num_labels):
    model = ResNetForImageClassification.from_pretrained(
        'microsoft/resnet-50', num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    return model, processor


def build_vision_model(model_key, num_labels):
    if model_key in ('vit_base', 'vit_large'):
        return load_vit(VISION_MODELS[model_key], num_labels)
    elif model_key == 'deit_base':
        return load_deit(VISION_MODELS[model_key], num_labels)
    elif model_key == 'swin_base':
        return load_swin(VISION_MODELS[model_key], num_labels)
    elif model_key == 'convnext_base':
        return load_convnext(VISION_MODELS[model_key], num_labels)
    raise ValueError(f"Unknown vision model: {model_key}")


def get_vision_model_type(model_key):
    mapping = {
        'vit_base': 'vit',
        'vit_large': 'vit',
        'deit_base': 'deit',
        'swin_base': 'swin',
        'convnext_base': 'convnext',
    }
    return mapping.get(model_key, 'vit')
