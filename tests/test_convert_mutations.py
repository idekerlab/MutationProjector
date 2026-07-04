import numpy as np
import pytest
import torch
from itertools import product

from nn_training_functions import convert_mutations
from reference_impls import old_convert_mutations


def test_bit_order_assumption():
    """Locks in the assumption the vectorized formula depends on: itertools.product's
    enumeration order over binary tuples IS binary counting order, MSB (leftmost) first."""
    assert list(product([0, 1], repeat=3))[5] == (1, 0, 1)
    assert list(product([0, 1], repeat=4))[9] == (1, 0, 0, 1)


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
