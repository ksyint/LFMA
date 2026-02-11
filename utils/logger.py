import os
import time
from torch.utils.tensorboard import SummaryWriter



class TrainingLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_time = time.time()
        self.log_file = os.path.join(log_dir, 'training.log')

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_text(self, message):
        elapsed = time.time() - self.start_time
        line = f'[{elapsed:.1f}s] {message}'
        print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')

    def close(self):
        self.writer.close()


class FileLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def write(self, message):
        with open(self.filepath, 'a') as f:
            f.write(message + '\n')

    def log_epoch(self, epoch, metrics):
        parts = [f'Epoch {epoch}']
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f'{k}: {v:.4f}')
            else:
                parts.append(f'{k}: {v}')
        self.write(' | '.join(parts))
