import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.default_config import get_vlm_args
from models.vlm.vlm_models import build_vlm_model, get_vlm_model_type, get_vlm_target_modules
from models.adapters.adapter_utils import (
from data.vlm_datasets import build_vlm_dataset, build_vlm_dataloader
from utils.logger import TrainingLogger
from utils.checkpoint import save_checkpoint
    import os

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
    targets = get_vlm_target_modules(model_key)
    if method == 'lfma':
        freeze_base_model(model)
        replaced = apply_lfma_to_model(model, targets, alpha=args.alpha, top_k_ratio=args.top_k_ratio)
        print(f'LFMA applied to {replaced} layers')
    elif method == 'fourier_ft':
        freeze_base_model(model)
        replaced = apply_fourier_ft_to_model(model, targets, n_freq=args.n_freq, alpha=args.alpha)
        print(f'FourierFT applied to {replaced} layers')
    elif method == 'lora':
        freeze_base_model(model)
        replaced = apply_lora_to_model(model, targets, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f'LoRA applied to {replaced} layers')
    elif method == 'full_ft':
        unfreeze_all(model)
    return model


def train_clip(model, processor, train_loader, val_loader, args, device):
    log_dir = f'{args.log_dir}/{args.model}_{args.task}_{args.method}'
    save_dir = f'{args.save_dir}/{args.model}_{args.task}_{args.method}'
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger(log_dir)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images = batch['pixel_values'].to(device)
            labels = batch['label']
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            text_inputs = processor(
                text=[f"a photo of class {i}" for i in range(len(labels))],
                return_tensors="pt", padding=True
            ).to(device)
            outputs = model(pixel_values=images, input_ids=text_inputs['input_ids'],
                            attention_mask=text_inputs['attention_mask'])
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0)
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f'[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}')
        logger.log_scalar('train/loss', avg_loss, epoch)
    logger.close()


def train_generic_vlm(model, processor, train_loader, val_loader, args, device):
    log_dir = f'{args.log_dir}/{args.model}_{args.task}_{args.method}'
    save_dir = f'{args.save_dir}/{args.model}_{args.task}_{args.method}'
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger(log_dir)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f'[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}')
        logger.log_scalar('train/loss', avg_loss, epoch)
    save_checkpoint(model, optimizer, args.epochs - 1, 0.0,
                    os.path.join(save_dir, 'last.pth'))
    logger.close()


def main():
    parser = get_vlm_args()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, processor = build_vlm_model(args.model)
    model = apply_method(model, args.method, args.model, args)
    model = model.to(device)
    print(f'Trainable params: {count_trainable_params(model):,} / Total: {count_total_params(model):,}')
    train_dataset = build_vlm_dataset(args.task, processor, split='train')
    val_dataset = build_vlm_dataset(args.task, processor, split='test')
    train_loader = build_vlm_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_vlm_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.model == 'clip':
        train_clip(model, processor, train_loader, val_loader, args, device)
    else:
        train_generic_vlm(model, processor, train_loader, val_loader, args, device)


if __name__ == '__main__':
    main()
