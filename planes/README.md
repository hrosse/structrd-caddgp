# Experiments: Planes in Satellite Imagery

## Setup

The dataset can be found on [kaggle](https://www.kaggle.com/datasets/rhammell/planesnet).
Copy the folder that contains the .png images to `/data` and from there run:
```console
python planes.py
```
The dataset is split into 75 percent training and 25 percent test data.

## Experiment 1

### Training

```console

python planes_train.py --model isotropic --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

python planes_train.py --model ARD --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.0015

python planes_train.py --model ACRD --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

python planes_train.py --model cadd --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

python planes_train.py --model APRD4x4ACRD --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

python planes_train.py --model caddAPRD4x4 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 350000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

```

### Evaluation

```console

python planes_eval.py --save_dir ./checkpoints/planes/isotropic --batch_size 2000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/ARD --batch_size 2000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/ACRD --batch_size 2000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/APRD4x4ACRD --batch_size 2000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/cadd --batch_size 2000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/caddAPRD4x4 --batch_size 2000 --verbose

```

## Experiment 2

### Training

```console

python planes_train.py --model Conv --M 750 --mb_size 240 --checkpoint_interval 1000 --max_steps 600000 --max_time 86400 --lr 0.01 --lr_decay_steps 7000 --lr_decay_factor 0.93 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.0001

python planes_train.py --model ConvACRD --M 750 --mb_size 240 --checkpoint_interval 1000 --max_steps 600000 --max_time 86400 --lr 0.01 --lr_decay_steps 7000 --lr_decay_factor 0.93 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.0001

python planes_train.py --model ConvRBFARD --M 750 --mb_size 240 --checkpoint_interval 1000 --max_steps 600000 --max_time 86400 --lr 0.01 --lr_decay_steps 7000 --lr_decay_factor 0.93 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.002

python planes_train.py --model ConvACRDcaddAPRD4x4 --M 750 --mb_size 240 --checkpoint_interval 1000 --max_steps 600000 --max_time 86400 --lr 0.01 --lr_decay_steps 7000 --lr_decay_factor 0.93 --verbose --n_train -1 --save_dir ./checkpoints/planes --default_jitter 0.001

```

### Evaluation

```console

python planes_eval.py --save_dir ./checkpoints/planes/Conv --batch_size 1000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/ConvACRD --batch_size 1000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/ConvRBFARD --batch_size 1000 --verbose

python planes_eval.py --save_dir ./checkpoints/planes/ConvACRDcaddAPRD4x4 --batch_size 1000 --verbose

```

## Accessing Results

After running the evaluation scripts the results are stored in `/results`.
The directory contains a notebook with code for loading the results after running evaluation scripts and for creating the figure of evidence lower bound, classification error and negative log predictive probability with respect to optimisation step.
