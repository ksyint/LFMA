import torch
import torch.nn as nn
from transformers import (

    RobertaForSequenceClassification,
    RobertaTokenizer,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    OPTForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)


LLM_MODELS = {
    'roberta_base': 'roberta-base',
    'roberta_large': 'roberta-large',
    'gpt2': 'gpt2',
    'opt_350m': 'facebook/opt-350m',
}


def load_roberta(model_name, num_labels):
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_gpt2(model_name, num_labels):
    model = GPT2ForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_opt(model_name, num_labels):
    model = OPTForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_llm_causal(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_llm_model(model_key, num_labels=2, task='classification'):
    if task == 'causal':
        return load_llm_causal(LLM_MODELS[model_key])
    if model_key in ('roberta_base', 'roberta_large'):
        return load_roberta(LLM_MODELS[model_key], num_labels)
    elif model_key == 'gpt2':
        return load_gpt2(LLM_MODELS[model_key], num_labels)
    elif model_key == 'opt_350m':
        return load_opt(LLM_MODELS[model_key], num_labels)
    raise ValueError(f"Unknown LLM model: {model_key}")


def get_llm_model_type(model_key):
    mapping = {
        'roberta_base': 'roberta',
        'roberta_large': 'roberta',
        'gpt2': 'gpt2',
        'opt_350m': 'opt',
    }
    return mapping.get(model_key, 'roberta')


def get_llm_target_modules(model_key):
    targets = {
        'roberta_base': ['query', 'value'],
        'roberta_large': ['query', 'value'],
        'gpt2': ['c_attn'],
        'opt_350m': ['q_proj', 'v_proj'],
    }
    return targets.get(model_key, ['query', 'value'])
