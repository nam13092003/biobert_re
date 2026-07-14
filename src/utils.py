import random
import os
import numpy as np
import torch

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory_summary():
    """Returns a string describing the current GPU memory usage."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    summary = []
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        summary.append(
            f"GPU {i} ({device_name}): Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB"
        )
    return " | ".join(summary)
