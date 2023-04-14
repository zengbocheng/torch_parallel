"""
@author: bochengz
@date: 2023/04/14
@email: bochengzeng@bochengz.top
"""
import torch


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is None:
        return model
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer


def save_checkpoint(model, optimizer, checkpoint_path):
    torch.save(
        {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        },
        checkpoint_path
    )

