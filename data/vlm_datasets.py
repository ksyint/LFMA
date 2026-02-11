import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset



class CLIPClassificationDataset(Dataset):
    def __init__(self, hf_dataset, processor, class_names, image_key='image'):
        self.dataset = hf_dataset
        self.processor = processor
        self.class_names = class_names
        self.image_key = image_key
        self.text_inputs = processor(
            text=[f"a photo of a {c}" for c in class_names],
            return_tensors="pt", padding=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt")
        label = item.get('label', item.get('fine_label', 0))
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'label': label,
        }


class VQADataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=40):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        question = item['question']
        encoding = self.processor(
            image, question, return_tensors="pt",
            padding='max_length', max_length=self.max_length, truncation=True
        )
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        if 'label' in item:
            encoding['labels'] = torch.tensor(item['label'])
        return encoding


class CaptioningDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=128, image_key='image', text_key='text'):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.image_key = image_key
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        text = item.get(self.text_key, '')
        if isinstance(text, list):
            text = text[0] if text else ''
        encoding = self.processor(
            images=image, text=text, return_tensors="pt",
            padding='max_length', max_length=self.max_length, truncation=True
        )
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        return encoding


def build_vlm_dataset(task, processor, split='train'):
    if task == 'clip_cifar100':
        ds = load_dataset('cifar100', split=split, trust_remote_code=True)
        class_names = ds.features['fine_label'].names
        return CLIPClassificationDataset(ds, processor, class_names, image_key='img')
    elif task == 'vqa_vilt':
        ds = load_dataset('Graphcore/vqa', split=split, trust_remote_code=True)
        return VQADataset(ds, processor)
    elif task == 'caption_blip':
        ds = load_dataset('nlphuji/flickr30k', split=split, trust_remote_code=True)
        return CaptioningDataset(ds, processor, image_key='image', text_key='caption')
    raise ValueError(f"Unknown VLM task: {task}")


def collate_fn_vlm(batch):
    elem = batch[0]
    result = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


def build_vlm_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn_vlm, num_workers=num_workers, pin_memory=True
    )
