import pytest
import torch

from device_utils import resolve_device, to_device_safe


def test_legacy_int_resolves_to_cuda_index():
    assert resolve_device(0) == torch.device('cuda:0')
    assert resolve_device(3) == torch.device('cuda:3')


def test_to_device_safe_preserves_float64_on_cpu():
    out = to_device_safe(torch.zeros(3, dtype=torch.float64), 'cpu')
    assert out.dtype == torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_to_device_safe_downcasts_float64_for_mps():
    out = to_device_safe(torch.zeros(3, dtype=torch.float64), 'mps')
    assert out.dtype == torch.float32
    assert out.device.type == 'mps'
