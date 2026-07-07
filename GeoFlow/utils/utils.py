import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import od_collate_fn


_DATALOADER_GENERATOR = None


def set_seed(seed):
    global _DATALOADER_GENERATOR
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _DATALOADER_GENERATOR = torch.Generator()
    _DATALOADER_GENERATOR.manual_seed(seed)


def build_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=od_collate_fn,
        generator=_DATALOADER_GENERATOR if shuffle else None,
    )


def lst_to_device(tensor_list, device, dtype=None):
    return [t.to(device=device, dtype=dtype) for t in tensor_list]


def pack_list_to_device(adj_pack_list, device, dtype=None):
    out = []
    for pack in adj_pack_list:
        out.append({
            k: (torch.as_tensor(v, device=device, dtype=dtype) if not torch.is_tensor(v)
                else v.to(device=device, dtype=dtype))
            for k, v in pack.items()
        })
    return out
