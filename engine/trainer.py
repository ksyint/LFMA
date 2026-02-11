import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.logger import TrainingLogger
from utils.checkpoint import save_checkpoint



class ImageTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 train_loader, val_loader, max_epochs=100,
                 log_dir='logs', save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.criterion = nn.CrossEntropyLoss()
        self.logger = TrainingLogger(log_dir)
        self.best_acc = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.max_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            self.logger.log_scalar('train/loss', train_loss, epoch)
            self.logger.log_scalar('train/acc', train_acc, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.val_loader is not None:
                val_loss, val_acc = self._validate(epoch)
                self.logger.log_scalar('val/loss', val_loss, epoch)
                self.logger.log_scalar('val/acc', val_acc, epoch)
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    save_checkpoint(self.model, self.optimizer, epoch, self.best_acc,
                                    os.path.join(self.save_dir, 'best.pth'))
                print(f'[Epoch {epoch+1}/{self.max_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        self.logger.close()

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
            inputs = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss / len(self.train_loader), 100.0 * correct / total

    def _validate(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return total_loss / len(self.val_loader), 100.0 * correct / total


class LLMTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 train_loader, val_loader, max_epochs=10,
                 log_dir='logs', save_dir='checkpoints', task_type='classification'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.task_type = task_type
        self.logger = TrainingLogger(log_dir)
        self.best_metric = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch(epoch)
            self.logger.log_scalar('train/loss', train_loss, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.val_loader is not None:
                val_loss, val_metric = self._validate(epoch)
                self.logger.log_scalar('val/loss', val_loss, epoch)
                self.logger.log_scalar('val/metric', val_metric, epoch)
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    save_checkpoint(self.model, self.optimizer, epoch, self.best_metric,
                                    os.path.join(self.save_dir, 'best.pth'))
                print(f'[Epoch {epoch+1}/{self.max_epochs}] '
                      f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}')
            else:
                print(f'[Epoch {epoch+1}/{self.max_epochs}] Train Loss: {train_loss:.4f}')
        self.logger.close()

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                if self.task_type == 'classification':
                    logits = outputs.logits
                    _, predicted = logits.max(1)
                    labels = batch['labels']
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        avg_loss = total_loss / len(self.val_loader)
        metric = 100.0 * correct / max(total, 1) if self.task_type == 'classification' else 0.0
        return avg_loss, metric
