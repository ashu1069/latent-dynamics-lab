import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True


def kl_balance(
    kl: torch.Tensor,
    free_bits: float = 1.0,
    balance: float = 0.8,
) -> torch.Tensor:
    """
    KL balancing: encourage posterior to use the latent space without
    letting KL explode. free_bits prevents posterior collapse.
    """
    kl = kl.clamp(min=free_bits)
    return balance * kl + (1 - balance) * kl.detach()
