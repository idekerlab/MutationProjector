"""
Item 3: proves the arithmetic replacement `idx = i * num_networks + j` is exactly what the old
pandas GATidx DataFrame lookup returned, for every (i, j) pair -- before that lookup structure
is even deleted from the codebase (it already has been; this test pins the invariant it relied on).
"""
from collections import defaultdict

import pandas as pd
import pytest


def build_old_gatidx(num_GATblock, num_networks):
    GATidx = defaultdict(list)
    count = 0
    for i in range(num_GATblock):
        for j in range(num_networks):
            GATidx['i'].append(i)
            GATidx['j'].append(j)
            GATidx['idx'].append(count)
            count += 1
    return pd.DataFrame(GATidx)


@pytest.mark.parametrize("num_GATblock,num_networks", [(1, 1), (2, 8), (3, 1), (1, 5), (4, 3)])
def test_arithmetic_formula_matches_pandas_lookup(num_GATblock, num_networks):
    GATidx = build_old_gatidx(num_GATblock, num_networks)
    for i in range(num_GATblock):
        for j in range(num_networks):
            old_idx = GATidx.loc[GATidx['i']==i,:].loc[GATidx['j']==j,:]['idx'].tolist()[0]
            new_idx = i * num_networks + j
            assert old_idx == new_idx
