# SR4MDL

> Official implementation of the paper: **Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length** (ICLR 2025), as well as its extended journal submission version:**An MDL-oriented Search Framework for Symbolic Regression** (submitting to TPAMI)

> NOTE: We are organizing the code and data for the extended version (An MDL-oriented Search Framework for Symbolic Regression), which will be updated in a week (before Sep. 18, 2025).

## Installation

Before starting, you may wanna create a virtual environment to avoid conflicts with other packages:
```bash
conda create --prefix ./venv python=3.12 -y
conda activate ./venv
```
Our code is based on nd2py library, which is a symbolic system written in pure Python. You can install it via pip or clone the repo:
```bash
# Install via pip
pip install git+https://github.com/yuzhTHU/nd2py

# Or clone the repo
git clone https://github.com/yuzhTHU/nd2py nd2py_package
pip install ./nd2py_package
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
python test.py --name demo --load_model ./results/train/demo/checkpoint.pth
```

## Symbolic Regression

To use the trained MDLformer model for symbolic regression, you have to:
1. Move the trained model to `./weights/checkpoint.pth`. (We provided a trained model in the Github release page as well as [Dropbox](https://www.dropbox.com/scl/fi/x1te3v1lmsrrr07r8uunr/checkpoint.pth?rlkey=v7ip8r6b4xuy4pdtk33jsyan5&st=iv36jfg2&dl=1))
2. Run the following command:
```bash
python ./demos/search_mcts4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "f=x1+x2*sin(x3)"
```
or
```bash
python ./demos/search_gp4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "f=x1+x2*sin(x3)"
```
The running result will be shown in the terminal, as well as saved in the `./results/search/` directory and `./results/aggregate.csv` file.

If you wanna test this model on Feynman & Strogatz dataset, you have to:
1. Install PMLB package from https://github.com/EpistasisLab/pmlb (`pip install pmlb` is not recommended since it does not contains these datasets, see https://epistasislab.github.io/pmlb/using-python.html)
```bash
cd data
git clone https://github.com/EpistasisLab/pmlb pmlb
pip install ./pmlb
cd ..
```
2. Run the following command:
```bash
python ./demos/search_mcts4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
or
```bash
python ./demos/search_gp4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
The running result will be shown in the terminal, as well as saved in the `./results/search/` directory and `./results/aggregate.csv` file.

## Run in SRBench

To test our method on the SRBench benchmark, you have to:

1. Clone the SRBench repo from [here](https://github.com/cavalab/srbench), save it to `./benchmark/srbench/` directory:
```bash
git clone https://github.com/cavalab/srbench ./benchmark/srbench/
```

3. Copy the contents of `methods` to `./benchmark/srbench/experiment/methods/`, and remember to replace `/path/to/weights/checkpoint.pth` with the path to the trained model.
```bash
cp ./methods/* ./benchmark/srbench/experiment/methods/
# Replace the /path/to/weights/checkpoint.pth to, for example, ./weights/checkpoint.pth
```
5. Run the following script starting from the root directory:
```bash
#!/bin/bash
method=mcts4mdl # or gp4mdl 
for seed in 29910 14423 28020 23654 15795 16850 21962 4426 5390 860; do
for noise in 0.000 0.001 0.01 0.1; do
for exp in strogatz_vdp2 feynman_I_6_2a strogatz_bacres2 strogatz_bacres1 feynman_II_27_18 feynman_II_3_24 feynman_I_6_2 feynman_II_8_31 feynman_I_12_1 feynman_I_12_5 feynman_I_14_4 feynman_I_39_1 strogatz_vdp1 feynman_I_25_13 feynman_I_26_2 feynman_I_29_4 strogatz_barmag1 feynman_II_11_28 feynman_II_38_14 strogatz_glider1 feynman_III_12_43 strogatz_shearflow2 strogatz_shearflow1 strogatz_predprey2 strogatz_barmag2 strogatz_predprey1 strogatz_lv2 strogatz_lv1 feynman_I_34_27 strogatz_glider2 feynman_I_12_4 feynman_III_17_37 feynman_I_43_31 feynman_I_14_3 feynman_III_15_27 feynman_I_15_10 feynman_I_16_6 feynman_I_18_12 feynman_I_39_11 feynman_III_15_14 feynman_III_15_12 feynman_II_13_34 feynman_II_13_23 feynman_I_27_6 feynman_II_10_9 feynman_I_30_3 feynman_I_30_5 feynman_I_37_4 feynman_I_34_1 feynman_III_8_54 feynman_I_47_23 feynman_I_10_7 feynman_II_15_4 feynman_II_34_2 feynman_II_34_29a feynman_II_34_2a feynman_test_10 feynman_II_37_1 feynman_I_48_2 feynman_III_7_38 feynman_II_4_23 feynman_I_34_14 feynman_I_6_2b feynman_II_27_16 feynman_II_24_17 feynman_II_8_7 feynman_II_15_5 feynman_I_43_16 feynman_test_5 feynman_I_34_8 feynman_I_50_26 feynman_test_3 feynman_I_38_12 feynman_I_39_22 feynman_test_15 feynman_test_11 feynman_I_8_14 feynman_I_43_43 feynman_test_8 feynman_III_10_19 feynman_I_24_6 feynman_II_13_17 feynman_II_34_11 feynman_II_11_27 feynman_I_32_5 feynman_III_4_33 feynman_III_21_20 feynman_II_38_3 feynman_II_6_11 feynman_II_6_15b feynman_I_12_2 feynman_III_4_32 feynman_I_29_16 feynman_I_13_4 feynman_I_15_3t feynman_I_18_4 feynman_III_13_18 feynman_I_18_14 feynman_I_15_3x feynman_I_12_11 feynman_II_2_42 feynman_test_7 feynman_test_4 feynman_II_34_29b feynman_II_11_3 feynman_II_11_20 feynman_test_18 feynman_II_35_18 feynman_I_44_4 feynman_test_14 feynman_test_13 feynman_test_12 feynman_II_35_21 feynman_test_9 feynman_I_41_16 feynman_III_19_51 feynman_I_13_12 feynman_III_14_14 feynman_II_21_32 feynman_III_9_52 feynman_I_32_17 feynman_test_2 feynman_test_19 feynman_test_17 feynman_II_6_15a feynman_I_11_19 feynman_I_40_1 feynman_test_16 feynman_test_20 feynman_test_1 feynman_test_6 feynman_II_36_38 feynman_I_9_18; do
    python ./benchmark/srbench/experiment/evaluate_model.py ./data/pmlb/datasets/$exp/$exp.tsv.gz \
        -ml $method \
        -seed $seed \
        -target_noise $noise \
        -results_path "./results-$method-$noise"
done
done
done
```
Or
```bash
for dataset in ./data/pmlb/datasets/strogatz_*/strogatz_*.tsv.gz; do
    python ./benchmark/srbench/experiment/analyze.py \
        "$dataset" \
        -time_limit 00:15 \
        -ml mcts4mdl \ # or gp4mdl
        -n_jobs 2 \
        -results ./results/srbench/ \
        -sym_data \
        --local \
        -script ./benchmark/srbench/experiment/evaluate_model
done

for dataset in ./data/pmlb/datasets/feynman_*/feynman_*.tsv.gz; do
    python ./benchmark/srbench/experiment/analyze.py \
        "$dataset" \
        -ml mcts4mdl \ # or gp4mdl
        --local \
        -n_jobs 2 \
        -results ./results/srbench/ \
        -script ./benchmark/srbench/experiment/evaluate_model \
        -sym_data
done
```
