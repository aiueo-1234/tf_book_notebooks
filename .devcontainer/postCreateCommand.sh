#!/bin/bash
pip install --upgrade pip && \
conda update -n base -c defaults conda && \
conda env create -f environment.yml && \
conda init && \
echo "conda activate book" >> ~/.bashrc

echo "export CONDA_DEFAULT_ENV=book" >> ~/.bashrc
echo "export PATH=/opt/conda/envs/book/bin:$PATH" >> ~/.bashrc

source ~/.bashrc
conda activate book
conda install -n book cudatoolkit cuda-version=11 -y
conda install -n book pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -n book transformers[tf,torch,sentencepiece,vision,optuna,sklearn,onnxruntime] accelerate pyarrow-all matplotlib ipywidgets umap-learn -c conda-forge -y
conda install -n book -c huggingface -c conda-forge datasets -y
conda install -n book tf-keras -y
cd ~/workspace/notebooks
pip install -r requirements.txt