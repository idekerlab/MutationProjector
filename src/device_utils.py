import torch


def resolve_device(device):
    """
    Normalizes a device spec to a torch.device. Accepts a plain int (legacy CUDA device
    index, e.g. 0 -> cuda:0, matching this codebase's historical `.cuda(0)` convention) for
    backward compatibility, or any device string/torch.device PyTorch accepts natively --
    'cuda:0', 'mps' (Apple Silicon GPU / unified memory), 'cpu', etc.
    """
    if isinstance(device, int):
        return torch.device(f'cuda:{device}')
    return torch.device(device)
