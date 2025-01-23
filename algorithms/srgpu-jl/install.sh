#!/bin/bash

# Check if running in correct environment
if [ "$CONDA_DEFAULT_ENV" != "srbench-srgpu.jl" ]; then
    echo "Please run the following commands first:"
    echo ">>> conda env create -f environment.yml"
    echo ">>> conda activate srbench-srgpu.jl"
    echo "Then run this script again."
    exit 1
fi

# remove directory if it exists
if [ -d SymbolicRegressionGPU.jl ]; then
    rm -rf SymbolicRegressionGPU.jl
fi

# clone repository
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl

# change directory
cd SymbolicRegressionGPU.jl

# install and build package
julia --project=. -e 'using Pkg; Pkg.develop(path="."); Pkg.build()'