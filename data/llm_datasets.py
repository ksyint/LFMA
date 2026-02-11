import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset



GLUE_TASKS = {
    'sst2': {'name': 'glue', 'subset': 'sst2', 'num_labels': 2, 'text_keys': ['sentence'], 'metric': 'accuracy'},
    'mrpc': {'name': 'glue', 'subset': 'mrpc', 'num_labels': 2, 'text_keys': ['sentence1', 'sentence2'], 'metric': 'accuracy'},
    'qnli': {'name': 'glue', 'subset': 'qnli', 'num_labels': 2, 'text_keys': ['question', 'sentence'], 'metric': 'accuracy'},
    'rte': {'name': 'glue', 'subset': 'rte', 'num_labels': 2, 'text_keys': ['sentence1', 'sentence2'], 'metric': 'accuracy'},
    'cola': {'name': 'glue', 'subset': 'cola', 'num_labels': 2, 'text_keys': ['sentence'], 'metric': 'mcc'},
    'stsb': {'name': 'glue', 'subset': 'stsb', 'num_labels': 1, 'text_keys': ['sentence1', 'sentence2'], 'metric': 'pcc'},
}


class GLUEDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, task_info, max_length=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.task_info = task_info
        self.max_length = max_length
        self.text_keys = task_info['text_keys']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(self.text_keys) == 1:
            encoding = self.tokenizer(
                item[self.text_keys[0]],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                item[self.text_keys[0]],
                item[self.text_keys[1]],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        result = {k: v.squeeze(0) for k, v in encoding.items()}
        label = item['label']
        if self.task_info['num_labels'] == 1:
            result['labels'] = torch.tensor(label, dtype=torch.float)
        else:
            result['labels'] = torch.tensor(label, dtype=torch.long)
        return result


class TextGenerationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=256, text_key='text'):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_key]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        result = {k: v.squeeze(0) for k, v in encoding.items()}
        result['labels'] = result['input_ids'].clone()
        return result


def build_glue_dataset(task_key, tokenizer, split='train', max_length=128):
    task_info = GLUE_TASKS[task_key]
    ds = load_dataset(task_info['name'], task_info['subset'], split=split, trust_remote_code=True)
    return GLUEDataset(ds, tokenizer, task_info, max_length), task_info


def build_text_gen_dataset(dataset_name, tokenizer, split='train', max_length=256):
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    return TextGenerationDataset(ds, tokenizer, max_length)


def collate_fn_llm(batch):
    elem = batch[0]
    result = {}
    for key in elem:
        result[key] = torch.stack([b[key] for b in batch])
    return result


def build_llm_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn_llm, num_workers=num_workers, pin_memory=True
    )
