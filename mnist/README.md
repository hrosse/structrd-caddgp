# Experiments: MNIST

## Setup

To set up the dataset place the MNIST dataset files in `/data` and from there run:
```console
python mnist.py
```

## Experiment 1: MNIST 6000

For this experiment a subset of 6000 points sampled from the original training data was used as training data.

### Training

```console

python mnist_train.py --model isotropic --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 180000 --max_time 60000 --lr 0.01 --lr_decay_steps 4000 --lr_decay_factor 0.9 --verbose --n_train 6000 --save_dir ./checkpoints/mnist6000

python mnist_train.py --model ARD --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 180000 --max_time 60000 --lr 0.01 --lr_decay_steps 4000 --lr_decay_factor 0.9 --verbose --n_train 6000 --save_dir ./checkpoints/mnist6000

python mnist_train.py --model APRD7x7 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 180000 --max_time 60000 --lr 0.01 --lr_decay_steps 4000 --lr_decay_factor 0.9 --verbose --n_train 6000 --save_dir ./checkpoints/mnist6000

python mnist_train.py --model APRD4x4 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 180000 --max_time 60000 --lr 0.01 --lr_decay_steps 4000 --lr_decay_factor 0.9 --verbose --n_train 6000 --save_dir ./checkpoints/mnist6000

python mnist_train.py --model APRD2x2 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 180000 --max_time 60000 --lr 0.01 --lr_decay_steps 4000 --lr_decay_factor 0.9 --verbose --n_train 6000 --save_dir ./checkpoints/mnist6000

```

### Evaluation

```console

python mnist_eval.py --save_dir ./checkpoints/mnist6000/isotropic --batch_size 2000 --verbose --result_dir ./results/mnist6000

python mnist_eval.py --save_dir ./checkpoints/mnist6000/ARD --batch_size 2000 --verbose --result_dir ./results/mnist6000

python mnist_eval.py --save_dir ./checkpoints/mnist6000/APRD7x7 --batch_size 2000 --verbose --result_dir ./results/mnist6000

python mnist_eval.py --save_dir ./checkpoints/mnist6000/APRD4x4 --batch_size 2000 --verbose --result_dir ./results/mnist6000

python mnist_eval.py --save_dir ./checkpoints/mnist6000/APRD2x2 --batch_size 2000 --verbose --result_dir ./results/mnist6000

```

## Experiment 2: full MNIST

For this experiment the full set of training data was used.

### Training

```console

python mnist_train.py --model isotropic --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 300000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/mnist60000

python mnist_train.py --model ARD --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 300000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/mnist60000

python mnist_train.py --model APRD7x7 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 300000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/mnist60000

python mnist_train.py --model APRD4x4 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 300000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/mnist60000

python mnist_train.py --model APRD2x2 --M 750 --mb_size 500 --checkpoint_interval 1000 --max_steps 300000 --max_time 72000 --lr 0.01 --lr_decay_steps 5000 --lr_decay_factor 0.9 --verbose --n_train -1 --save_dir ./checkpoints/mnist60000

```

### Evaluation

```console

python mnist_eval.py --save_dir ./checkpoints/mnist60000/isotropic --batch_size 2000 --verbose --result_dir ./results/mnist60000

python mnist_eval.py --save_dir ./checkpoints/mnist60000/ARD --batch_size 2000 --verbose --result_dir ./results/mnist60000

python mnist_eval.py --save_dir ./checkpoints/mnist60000/APRD7x7 --batch_size 2000 --verbose --result_dir ./results/mnist60000

python mnist_eval.py --save_dir ./checkpoints/mnist60000/APRD4x4 --batch_size 2000 --verbose --result_dir ./results/mnist60000

python mnist_eval.py --save_dir ./checkpoints/mnist60000/APRD2x2 --batch_size 2000 --verbose --result_dir ./results/mnist60000

```

## Accessing Results

After running the evaluation scripts the results are stored in `/results`.
The directory contains a notebook with code for loading the results after running evaluation scripts and for creating the figure of evidence lower bound, classification error and negative log predictive probability with respect to optimisation step.


