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


def to_device_safe(tensor, device):
    """
    Moves `tensor` to `device`, downcasting float64 -> float32 first if the target device is
    'mps' (Apple Silicon), which doesn't support float64 at all and raises rather than
    downcasting automatically. No-op dtype change on any other device (cpu/cuda keep full
    float64 precision if the input already has it) -- this only touches behavior on MPS, where
    some precision loss is an unavoidable cost of using that backend at all, not a choice made
    here. Typical trigger: raw input data loaded via pandas (which defaults numeric columns to
    float64) passed straight into a model running on MPS.
    """
    device = resolve_device(device)
    if device.type == 'mps' and tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    return tensor.to(device)
