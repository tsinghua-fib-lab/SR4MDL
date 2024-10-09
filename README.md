# SR4MDL

> Official implementation of the paper: **Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length**.

## Installation

We implement our method based on Python3.12.7. To install the package, you can run the following command:
```bash
pip install sr4mdl
```

## Train

To train the MDLformer model, you can run the following command:
```bash
python train.py --name demo
```
It will train the model on the synthetic dataset and save the model in the `./results/train/demo/` directory.

## Test

To test the trained MDLformer model, you can run the following command:
```bash
python test.py --name demo --load_model ./results/train/demo/model.pth
```

## Symbolic Regression

To use the trained MDLformer model for symbolic regression, you have to:
1. Move the trained model to `./weights/checkpoint.pth`. (We provided a trained model in the release page)
2. Run the following command:
```bash
python search.py --load_model ./weights/checkpoint.pth --name demo --function "f=x1+x2*sin(x3)"
```
The running result will be shown in the terminal, as well as saved in the `./results/search/demo/` directory and `./results/aggregate.csv` file.

If you wanna test this model on Feynman & Strogatz dataset, you have to:
1. Install PMLB package from https://github.com/EpistasisLab/pmlb (`pip install pmlb` is not recommended since it does not contains these datasets, see https://epistasislab.github.io/pmlb/using-python.html)
```bash
cd data
git clone https://github.com/EpistasisLab/pmlb pmlb
cd pmlb/
pip install .
cd ..
```
2. Run the following command:
```bash
python search.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
The running result will be shown in the terminal, as well as saved in the `./results/search/demo/` directory and `./results/aggregate.csv` file.
