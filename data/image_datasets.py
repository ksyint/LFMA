import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
from datasets import load_dataset

    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor,
)


IMAGE_DATASETS = {
    'cifar10': {'name': 'cifar10', 'num_classes': 10, 'image_key': 'img', 'label_key': 'label'},
    'cifar100': {'name': 'cifar100', 'num_classes': 100, 'image_key': 'img', 'label_key': 'fine_label'},
    'oxford_pets': {'name': 'timm/oxford-iiit-pet', 'num_classes': 37, 'image_key': 'image', 'label_key': 'label'},
    'fgvc_aircraft': {'name': 'Donghyun99/FGVC-Aircraft', 'num_classes': 100, 'image_key': 'image', 'label_key': 'label'},
    'eurosat': {'name': 'timm/eurosat-rgb', 'num_classes': 10, 'image_key': 'image', 'label_key': 'label'},
}


def get_image_transforms(processor, is_train=True):
    size = processor.size.get('height', 224) if hasattr(processor, 'size') else 224
    normalize = Normalize(
        mean=processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406],
        std=processor.image_std if hasattr(processor, 'image_std') else [0.229, 0.224, 0.225],
    )
    if is_train:
        return Compose([
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
    return Compose([
        Resize(size + 32),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])


def build_image_dataset(dataset_key, processor, split='train'):
    ds_info = IMAGE_DATASETS[dataset_key]
    is_train = split == 'train'
    transforms = get_image_transforms(processor, is_train=is_train)
    image_key = ds_info['image_key']
    label_key = ds_info['label_key']

    def preprocess(batch):
        batch['pixel_values'] = [transforms(img.convert('RGB')) for img in batch[image_key]]
        batch['labels'] = batch[label_key]
        return batch

    dataset = load_dataset(ds_info['name'], split=split, trust_remote_code=True)
    dataset.set_transform(preprocess)
    return dataset, ds_info['num_classes']


def collate_fn_image(batch):
    pixel_values = torch.stack([example['pixel_values'] for example in batch])
    labels = torch.tensor([example['labels'] for example in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


def build_image_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn_image, num_workers=num_workers, pin_memory=True
    )
