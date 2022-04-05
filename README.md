# Transformer-Pytorch
A pytorch implementations for Transformer

The Transformer paper:https://arxiv.org/abs/1706.03762

The introduction to Transformer can be found in [link](https://wmathor.com/index.php/archives/1438/)

Code reference:https://wmathor.com/index.php/archives/1455/
# 1. Settings
Using these steps to prepare.

All dependencies can be installed into a conda environment with the provided environment.yaml file.
## 1.1 Clone the repository
```
git clone https://github.com/EightBarren7/Transformer-Pytorch.git
cd Transformer-Pytorch
```
## 1.2 Create conda environment and activate
```
conda env create -f environment.yaml
conda activate Pytorch-Transformer
```
# 2. Usage
## 2.1 Adjust the hyper-parameters
You can change the **batch_size** according to your GPU.

If you want train a new model, check the ```config.py```, and keep the **checkpoint_path** none. It will save the best model during training.

If you want to continue training a model, edit the **checkpoint_path** to your model path.
## 2.2 Training
Using ```python train.py``` to start training.
## 2.3 Testing
Edit the **checkpoint_path** to your model path and then using ```python translation``` to test your model.

A model trained after 100 epochs is provided to _result_ directory.
