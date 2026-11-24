import numpy as np
import pytest
import torch

from nn_training_functions import convert_mutations
from reference_impls import old_convert_mutations


@pytest.mark.parametrize("trial", range(20))
def test_matches_reference_loop(trial):
    rng = np.random.RandomState(trial)
    N = rng.randint(1, 30)
    G = rng.randint(1, 50)
    k = rng.choice([1, 2, 3, 4])
    X = torch.from_numpy(rng.randint(0, 2, size=(N, G, k)))

    old = old_convert_mutations(X)
    new = convert_mutations(X)
    assert torch.equal(old, new)
