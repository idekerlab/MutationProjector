#!/bin/bash

pushd /opt/MutationProjector > /dev/null

/opt/conda/envs/MutationProjector/bin/python src/predict.py "$@"

popd > /dev/null 2&>1
