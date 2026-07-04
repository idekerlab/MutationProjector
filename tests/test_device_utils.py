"""
device_utils.resolve_device: normalizes the legacy `.cuda(int)` convention (an int device
index) and any native torch device spec (string or torch.device) -- including 'mps' for
Apple Silicon's GPU/unified memory -- to a torch.device. Constructing a torch.device('mps')
doesn't require an actual MPS backend to be present (only *using* it would), so this is
testable in any environment.
"""
import torch

from device_utils import resolve_device, to_device_safe


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


def test_to_device_safe_preserves_float64_on_cpu():
    # cpu/cuda support float64 fine -- to_device_safe must not change dtype there, or it would
    # silently diverge from the bit-exact CPU/CUDA equivalence already verified elsewhere.
    t = torch.zeros(3, dtype=torch.float64)
    out = to_device_safe(t, 'cpu')
    assert out.dtype == torch.float64
    assert out.device == torch.device('cpu')


def test_to_device_safe_preserves_float32():
    t = torch.zeros(3, dtype=torch.float32)
    out = to_device_safe(t, 'cpu')
    assert out.dtype == torch.float32


def test_to_device_safe_downcasts_float64_for_mps(monkeypatch):
    # This sandbox has no real MPS backend, so the final device transfer is monkeypatched to a
    # no-op -- this test only verifies the dtype-downcast *decision*, not an actual MPS move.
    calls = []
    real_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        calls.append(self.dtype)
        if args and isinstance(args[0], torch.device) and args[0].type == 'mps':
            return self  # pretend the (unavailable) MPS move succeeded
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', fake_to)

    t = torch.zeros(3, dtype=torch.float64)
    to_device_safe(t, 'mps')
    # the dtype cast (float64 -> float32) must happen before the device transfer is attempted
    assert calls[0] == torch.float64  # the .to(torch.float32) cast call sees the original dtype
    assert calls[-1] == torch.float32  # the .to(device) call happens on the already-downcast tensor
