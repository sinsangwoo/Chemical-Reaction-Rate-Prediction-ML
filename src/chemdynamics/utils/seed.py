import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def set_deterministic_seed(seed: int = 42) -> None:
    """Ensure deterministic reproducibility across all modules."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
