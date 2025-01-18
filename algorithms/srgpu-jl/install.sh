# !/bin/bash
# install ITEA

# remove directory if it exists
if [ -d srgpu-jl ]; then
    rm -rf srgpu-jl
fi

# git clone https://github.com/folivetti/tir.git
git clone https://github.com/x66ccff/SymbolicRegressionGPU.jl

# cd tir
cd SymbolicRegressionGPU.jl
# git checkout 9f51131cabdd52d4dab7a00cbd425bbae05b15b9

chmod +x install.sh
./install.sh

# WGL NOTE: this is a temp fix until PR https://github.com/folivetti/ITEA/pull/12 is merged
# install ghcup
# export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
# curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
# export PATH=$PATH:~/.ghcup/bin:~/.cabal/bin 

# #conda activate srbench
# cabal install --overwrite-policy=always --installdir=./python && cd python && pip install .
