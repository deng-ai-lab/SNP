# Score-based Neural Processes

This repository is the official implementation of Score-based Neural Processes (SNPs). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train_snp.py --dataset celeba32

python train_ERA5.py 

python train_CFD.py 
```


## Evaluation

To evaluate my model, run:

```eval
python test_snp.py --dataset celeba32

python test_ERA5.py 

python test_CFD.py 
```
