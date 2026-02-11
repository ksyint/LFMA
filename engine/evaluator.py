import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score



class ImageEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                inputs = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100.0 * correct / total
        return {'accuracy': accuracy, 'predictions': all_preds, 'labels': all_labels}


class LLMEvaluator:
    def __init__(self, model, device, task_metric='accuracy'):
        self.model = model
        self.device = device
        self.task_metric = task_metric

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                logits = outputs.logits
                if self.task_metric in ('accuracy', 'mcc'):
                    preds = logits.argmax(dim=-1)
                else:
                    preds = logits.squeeze(-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        results = {'loss': avg_loss}
        if self.task_metric == 'accuracy':
            results['accuracy'] = accuracy_score(all_labels, all_preds) * 100
        elif self.task_metric == 'mcc':
            results['mcc'] = matthews_corrcoef(all_labels, all_preds)
        elif self.task_metric == 'pcc':
            results['pcc'] = float(np.corrcoef(all_labels, all_preds)[0, 1])
        return results


class CLIPEvaluator:
    def __init__(self, model, processor, device, class_names):
        self.model = model
        self.processor = processor
        self.device = device
        self.class_names = class_names

    def evaluate(self, dataloader):
        self.model.eval()
        text_inputs = self.processor(
            text=[f"a photo of a {c}" for c in self.class_names],
            return_tensors="pt", padding=True
        ).to(self.device)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='CLIP Eval'):
                images = batch['pixel_values'].to(self.device)
                labels = batch['label'] if isinstance(batch['label'], torch.Tensor) else torch.tensor(batch['label'])
                labels = labels.to(self.device)
                outputs = self.model(pixel_values=images, **text_inputs)
                logits = outputs.logits_per_image
                preds = logits.argmax(dim=-1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        return {'accuracy': 100.0 * correct / max(total, 1)}
