#!/bin/bash

# Check if running in correct environment
if [ "$CONDA_DEFAULT_ENV" != "srbench-srgpu-jl" ]; then
    echo "Please run the following commands first:"
    echo ">>> conda env create -f environment.yml"
    echo ">>> conda activate srbench-srgpu-jl"
    echo "Then run this script again."
    exit 1
fi

# which julia # curl -fsSL https://install.julialang.org | sh -s -- -y
# command -v julia >/dev/null 2>&1 || { echo "Installing Julia..."; curl -fsSL https://install.julialang.org | sh -s -- -y; }
command -v julia >/dev/null 2>&1 || { echo "Installing Julia..."; curl -fsSL https://install.julialang.org | sh -s -- -y && source ~/.bashrc; }

# remove directory if it exists
if [ -d SymbolicRegressionGPU.jl ]; then
    rm -rf SymbolicRegressionGPU.jl
fi

# clone repository
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl

# change directory
cd SymbolicRegressionGPU.jl

# install and build package
julia -e 'using Pkg; Pkg.develop(path="."); Pkg.build(verbose=true)' # global julia
# julia --project=. -e 'using Pkg; Pkg.instantiate(path="."); Pkg.develop(path="."); Pkg.build()' # conda julia