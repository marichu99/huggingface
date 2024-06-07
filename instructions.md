# Create and activate the conda environment
conda create --name myenv python=3.9
conda activate myenv

# Install conda packages from conda-forge
conda install -c conda-forge tiktoken sentence-transformers langchain-core

# Install remaining packages using pip
pip install tf-keras langchain-community langchain chromadb
