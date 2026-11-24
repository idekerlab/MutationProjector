import numpy as np
import torch

from nn_training_functions import convert_mutations
from reference_impls import old_convert_mutations


def test_matches_reference_loop():
    failures = []
    for trial in range(20):
        rng = np.random.RandomState(trial)
        N = rng.randint(1, 30)
        G = rng.randint(1, 50)
        k = rng.choice([1, 2, 3, 4])
        X = torch.from_numpy(rng.randint(0, 2, size=(N, G, k)))
        if not torch.equal(old_convert_mutations(X), convert_mutations(X)):
            failures.append(dict(trial=trial, N=N, G=G, k=k))
    assert failures == []
