#!/bin/bash

# remove directory if it exists
if [ -d SymbolicRegressionGPU.jl ]; then
    rm -rf SymbolicRegressionGPU.jl
fi

# clone repository
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl

# change directory
cd SymbolicRegressionGPU.jl

# install and build package
# julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia -e 'using Pkg; Pkg.develop(path="."); Pkg.build()'