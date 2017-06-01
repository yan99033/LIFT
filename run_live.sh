#!/bin/bash

# ------------------------------------------------------------
# Example script for running LIFT

# Open MP Settings
export OMP_NUM_THREADS=1

# Cuda Settings
export CUDA_VISIBLE_DEVICES=0

# Theano Flags 
export THEANO_FLAGS="device=gpu0,${THEANO_FLAGS}"

# ------------------------------------------------------------
# LIFT code settings

# Number of keypoints
_LIFT_NUM_KEYPOINT=1000

# Whether to save debug image for keypoints
_LIFT_SAVE_PNG=1

# Whether the use Theano when keypoint testing. CuDNN is required when turned
# on
_LIFT_USE_THEANO=1

# The base path of the code
export _LIFT_BASE_PATH="$(pwd)"

_LIFT_PYTHON_CODE_PATH="${_LIFT_BASE_PATH}/python-code"


(cd $_LIFT_PYTHON_CODE_PATH; \
 python live.py
)


