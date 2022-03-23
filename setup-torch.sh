#!/usr/bin/env bash

# Downloaded torch C++ API extracted to openfoam-ml/libtorch

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export OF_TORCH=$SCRIPT_DIR/libtorch/include
export OF_TORCH_INCLUDE=$OF_TORCH/torch/csrc/api/include
export OF_TORCH_LIB=$SCRIPT_DIR/libtorch/lib

#  Local Arxh Linux system 

#export OF_TORCH=/usr/include/
#export OF_TORCH_INCLUDE=$OF_TORCH/torch/csrc/api/include/
#export OF_TORCH_LIB=/usr/lib

