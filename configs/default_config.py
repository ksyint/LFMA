import argparse



def get_image_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base',
                        choices=['vit_base', 'vit_large', 'deit_base', 'swin_base', 'convnext_base'])
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'oxford_pets', 'fgvc_aircraft', 'eurosat'])
    parser.add_argument('--method', type=str, default='lfma',
                        choices=['lfma', 'fourier_ft', 'lora', 'full_ft'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=12.0)
    parser.add_argument('--top_k_ratio', type=float, default=0.05)
    parser.add_argument('--n_freq', type=int, default=1000)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs/image')
    parser.add_argument('--save_dir', type=str, default='checkpoints/image')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser


def get_vlm_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='clip',
                        choices=['clip', 'blip_caption', 'blip2_caption', 'vilt_vqa'])
    parser.add_argument('--task', type=str, default='clip_cifar100',
                        choices=['clip_cifar100', 'vqa_vilt', 'caption_blip'])
    parser.add_argument('--method', type=str, default='lfma',
                        choices=['lfma', 'fourier_ft', 'lora', 'full_ft'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=12.0)
    parser.add_argument('--top_k_ratio', type=float, default=0.05)
    parser.add_argument('--n_freq', type=int, default=500)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs/vlm')
    parser.add_argument('--save_dir', type=str, default='checkpoints/vlm')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser


def get_llm_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='roberta_base',
                        choices=['roberta_base', 'roberta_large', 'gpt2', 'opt_350m'])
    parser.add_argument('--task', type=str, default='sst2',
                        choices=['sst2', 'mrpc', 'qnli', 'rte', 'cola', 'stsb'])
    parser.add_argument('--method', type=str, default='lfma',
                        choices=['lfma', 'fourier_ft', 'lora', 'full_ft'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=12.0)
    parser.add_argument('--top_k_ratio', type=float, default=0.0016)
    parser.add_argument('--n_freq', type=int, default=500)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs/llm')
    parser.add_argument('--save_dir', type=str, default='checkpoints/llm')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser
