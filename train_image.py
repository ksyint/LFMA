import random
import numpy as np
import torch
from configs.default_config import get_image_args
from models.vision.image_classifiers import build_vision_model, get_vision_model_type
from models.adapters.adapter_utils import (
from data.image_datasets import build_image_dataset, build_image_dataloader
from engine.trainer import ImageTrainer
    from data.image_datasets import IMAGE_DATASETS

    apply_lfma_to_model,
    apply_fourier_ft_to_model,
    apply_lora_to_model,
    freeze_base_model,
    unfreeze_all,
    count_trainable_params,
    count_total_params,
    get_adapter_target_modules,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_method(model, method, model_type, args):
    targets = get_adapter_target_modules(model_type)
    if method == 'lfma':
        freeze_base_model(model)
        replaced = apply_lfma_to_model(model, targets, alpha=args.alpha, top_k_ratio=args.top_k_ratio)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        print(f'LFMA applied to {replaced} layers')
    elif method == 'fourier_ft':
        freeze_base_model(model)
        replaced = apply_fourier_ft_to_model(model, targets, n_freq=args.n_freq, alpha=args.alpha)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        print(f'FourierFT applied to {replaced} layers')
    elif method == 'lora':
        freeze_base_model(model)
        replaced = apply_lora_to_model(model, targets, rank=args.lora_rank, alpha=args.lora_alpha)
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters():
                p.requires_grad = True
        print(f'LoRA applied to {replaced} layers')
    elif method == 'full_ft':
        unfreeze_all(model)
    return model


def main():
    parser = get_image_args()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    num_classes = IMAGE_DATASETS[args.dataset]['num_classes']
    model, processor = build_vision_model(args.model, num_classes)
    model_type = get_vision_model_type(args.model)
    model = apply_method(model, args.method, model_type, args)
    model = model.to(device)
    print(f'Trainable params: {count_trainable_params(model):,} / Total: {count_total_params(model):,}')
    train_split = 'train'
    val_split = 'test' if args.dataset in ('cifar10', 'cifar100') else 'test'
    train_dataset, _ = build_image_dataset(args.dataset, processor, split=train_split)
    val_dataset, _ = build_image_dataset(args.dataset, processor, split=val_split)
    train_loader = build_image_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_image_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log_dir = f'{args.log_dir}/{args.model}_{args.dataset}_{args.method}'
    save_dir = f'{args.save_dir}/{args.model}_{args.dataset}_{args.method}'
    trainer = ImageTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        train_loader=train_loader, val_loader=val_loader, max_epochs=args.epochs,
        log_dir=log_dir, save_dir=save_dir
    )
    trainer.train()


if __name__ == '__main__':
    main()
