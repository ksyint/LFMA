import torch
import torch.nn as nn
from transformers import (

    CLIPModel,
    CLIPProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)


VLM_MODELS = {
    'clip': 'openai/clip-vit-base-patch16',
    'blip_caption': 'Salesforce/blip-image-captioning-base',
    'blip2_caption': 'Salesforce/blip2-opt-2.7b',
    'vilt_vqa': 'dandelin/vilt-b32-finetuned-vqa',
}


def load_clip(model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def load_blip(model_name):
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    return model, processor


def load_blip2(model_name):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_vilt(model_name):
    model = ViltForQuestionAnswering.from_pretrained(model_name)
    processor = ViltProcessor.from_pretrained(model_name)
    return model, processor


def build_vlm_model(model_key):
    if model_key == 'clip':
        return load_clip(VLM_MODELS[model_key])
    elif model_key == 'blip_caption':
        return load_blip(VLM_MODELS[model_key])
    elif model_key == 'blip2_caption':
        return load_blip2(VLM_MODELS[model_key])
    elif model_key == 'vilt_vqa':
        return load_vilt(VLM_MODELS[model_key])
    raise ValueError(f"Unknown VLM model: {model_key}")


def get_vlm_model_type(model_key):
    mapping = {
        'clip': 'clip_vision',
        'blip_caption': 'blip',
        'blip2_caption': 'blip',
        'vilt_vqa': 'vit',
    }
    return mapping.get(model_key, 'vit')


def get_vlm_target_modules(model_key):
    targets = {
        'clip': ['q_proj', 'v_proj'],
        'blip_caption': ['query', 'value'],
        'blip2_caption': ['q_proj', 'v_proj'],
        'vilt_vqa': ['query', 'value'],
    }
    return targets.get(model_key, ['query', 'value'])
