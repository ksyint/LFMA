import os
import torch



def save_checkpoint(model, optimizer, epoch, best_metric, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def load_model_weights(filepath, model, strict=True):
    checkpoint = torch.load(filepath, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_adapter_weights(model, filepath, adapter_cls=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    adapter_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_state[name] = param.data
    torch.save(adapter_state, filepath)


def load_adapter_weights(model, filepath):
    adapter_state = torch.load(filepath, map_location='cpu')
    model_state = model.state_dict()
    model_state.update(adapter_state)
    model.load_state_dict(model_state, strict=False)
    return model
