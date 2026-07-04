"""
device_utils.resolve_device: normalizes the legacy `.cuda(int)` convention (an int device
index) and any native torch device spec (string or torch.device) -- including 'mps' for
Apple Silicon's GPU/unified memory -- to a torch.device. Constructing a torch.device('mps')
doesn't require an actual MPS backend to be present (only *using* it would), so this is
testable in any environment.
"""
import torch

from device_utils import resolve_device


def test_legacy_int_resolves_to_cuda_index():
    assert resolve_device(0) == torch.device('cuda:0')
    assert resolve_device(3) == torch.device('cuda:3')


def test_string_devices_pass_through():
    assert resolve_device('cpu') == torch.device('cpu')
    assert resolve_device('mps') == torch.device('mps')
    assert resolve_device('cuda:0') == torch.device('cuda:0')


def test_torch_device_passes_through():
    d = torch.device('cpu')
    assert resolve_device(d) == d
