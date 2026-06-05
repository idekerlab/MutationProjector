#!/bin/bash

set -euo pipefail

pushd /opt/MutationProjector/src > /dev/null

/opt/conda/envs/MutationProjector/bin/python predict.py "$@"

popd > /dev/null 2>&1
