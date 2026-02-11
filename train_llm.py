import random
import numpy as np
import torch
from configs.default_config import get_llm_args
from models.llm.llm_models import build_llm_model, get_llm_model_type, get_llm_target_modules
from models.adapters.adapter_utils import (
from data.llm_datasets import build_glue_dataset, build_llm_dataloader, GLUE_TASKS
from engine.trainer import LLMTrainer

    apply_lfma_to_model,
    apply_fourier_ft_to_model,
    apply_lora_to_model,
    freeze_base_model,
    unfreeze_all,
    count_trainable_params,
    count_total_params,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_method(model, method, model_key, args):
    targets = get_llm_target_modules(model_key)
    if method == 'lfma':
        freeze_base_model(model)
        replaced = apply_lfma_to_model(model, targets, alpha=args.alpha, top_k_ratio=args.top_k_ratio)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if hasattr(model, 'score'):
            for p in model.score.parameters():
                p.requires_grad = True
        print(f'LFMA applied to {replaced} layers')
    elif method == 'fourier_ft':
        freeze_base_model(model)
        replaced = apply_fourier_ft_to_model(model, targets, n_freq=args.n_freq, alpha=args.alpha)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if hasattr(model, 'score'):
            for p in model.score.parameters():
                p.requires_grad = True
        print(f'FourierFT applied to {replaced} layers')
    elif method == 'lora':
        freeze_base_model(model)
        replaced = apply_lora_to_model(model, targets, rank=args.lora_rank, alpha=args.lora_alpha)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if hasattr(model, 'score'):
            for p in model.score.parameters():
                p.requires_grad = True
        print(f'LoRA applied to {replaced} layers')
    elif method == 'full_ft':
        unfreeze_all(model)
    return model


def main():
    parser = get_llm_args()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    task_info = GLUE_TASKS[args.task]
    num_labels = task_info['num_labels']
    model, tokenizer = build_llm_model(args.model, num_labels=num_labels, task='classification')
    model_type = get_llm_model_type(args.model)
    model = apply_method(model, args.method, args.model, args)
    model = model.to(device)
    print(f'Trainable params: {count_trainable_params(model):,} / Total: {count_total_params(model):,}')
    train_dataset, _ = build_glue_dataset(args.task, tokenizer, split='train', max_length=args.max_length)
    val_dataset, _ = build_glue_dataset(args.task, tokenizer, split='validation', max_length=args.max_length)
    train_loader = build_llm_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_llm_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log_dir = f'{args.log_dir}/{args.model}_{args.task}_{args.method}'
    save_dir = f'{args.save_dir}/{args.model}_{args.task}_{args.method}'
    trainer = LLMTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        train_loader=train_loader, val_loader=val_loader, max_epochs=args.epochs,
        log_dir=log_dir, save_dir=save_dir, task_type='classification'
    )
    trainer.train()


if __name__ == '__main__':
    main()
