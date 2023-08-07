# Using Confounded Data in Latent Model-Based Reinforcement Learning

## Requirements

All experiments can be reproduced in a conda environment with the following dependencies:
```
python=3.10.8
pytorch=1.13.1
numpy=1.22.3
matplotlib=3.6.3
gzip
pickle
```

# Toy problem 1 (door)

This toy problem corresponds to the guiding example in the paper, and is provided for debugging purposes. It is configured in the following files:
```
experiments/toy1/config.json
experiments/toy1/hyperparams.json
```

To run this experiment execute the following commands:
```shell
EXPERIMENT = toy1

# decide on an expert (agent used to produce confounded offline data)
# one of 'noisy_good' 'perfect_good' 'perfect_bad' 'random' 'strong_bad_bias' 'strong_good_bias'
EXPERT = 'noisy_good'

# GPU ID (-1 for CPU)
GPU = 0

# run entire experiment (3 methods x 10 seeds)
for SEED in {0..9}; do
  for METHOD in int obs+int augmented; do
    bash scripts/run_exp.sh --experiment $EXPERIMENT --expert $EXPERT --method $METHOD --seed $SEED --gpu $GPU
  done
done

# produce likelihood and return plots
python scripts/07_plot.py --experiment $EXPERIMENT --expert $EXPERT --nseeds 10
```

Results are stored in the following folders:
```
experiments/
  toy1/
    plots/
    results/
```

# Toy problem 2 (tiger)

The `tiger` problem from the paper. It is configured in the following files:
```
experiments/toy2/config.json
experiments/toy2/hyperparams.json
```

To run this experiment execute the following commands:
```shell
EXPERIMENT = toy2

# decide on an expert (agent used to produce confounded offline data)
# one of 'noisy_good' 'perfect_good' 'perfect_bad' 'random'
EXPERT = 'noisy_good'

# GPU ID (-1 for CPU)
GPU = 0

# run entire experiment (3 methods x 10 seeds)
for SEED in {0..9}; do
  for METHOD in int obs+int augmented; do
    bash scripts/run_exp.sh --experiment $EXPERIMENT --expert $EXPERT --method $METHOD --seed $SEED --gpu $GPU
  done
done

# produce likelihood and return plots
python scripts/07_plot.py --experiment $EXPERIMENT --expert $EXPERT --nseeds 10
```

Results are stored in the following folders:
```
experiments/
  toy2/
    plots/
    results/
```

# Toy problem 3 (sloppy dark room)

The `sloppy dark room` problem from the paper. It is configured in the following files:
```
experiments/toy3/config.json
experiments/toy3/hyperparams.json
```

To run this experiment execute the following commands:
```shell
EXPERIMENT = toy3

# expert (agent used to produce confounded offline data)
EXPERT = 'shortest_path'

# GPU ID (-1 for CPU)
GPU = 0

# run entire experiment (3 methods x 10 seeds)
for SEED in {0..9}; do
  for METHOD in int obs+int augmented; do
    bash scripts/run_exp.sh --experiment $EXPERIMENT --expert $EXPERT --method $METHOD --seed $SEED --gpu $GPU
  done
done

# produce likelihood and return plots
python scripts/07_plot.py --experiment $EXPERIMENT --expert $EXPERT --nseeds 10

# produce trajectory density plots
python scripts/09_plot_grid_density.py --experiment $EXPERIMENT --expert $EXPERT --nseeds 10
```

Results are stored in the following folders:
```
experiments/
  toy3/
    plots/
    results/
```

# Toy problem 4 (hidden treasures)

The `hidden treasures` problem from the paper. It is configured in the following files:
```
experiments/toy4/config.json
experiments/toy4/hyperparams.json
```

To run this experiment execute the following commands:
```shell
EXPERIMENT = toy4

# expert (agent used to produce confounded offline data)
EXPERT = 'shortest_path'

# GPU ID (-1 for CPU)
GPU = 0

# run entire experiment (3 methods x 10 seeds)
for SEED in {0..9}; do
  for METHOD in int obs+int augmented; do
    bash scripts/run_exp.sh --experiment $EXPERIMENT --expert $EXPERT --method $METHOD --seed $SEED --gpu $GPU
  done
done

# produce likelihood and return plots
python scripts/07_plot.py --experiment $EXPERIMENT --expert $EXPERT --nseeds 10

# produce trajectory density plots (single seed)
python scripts/09_plot_grid_density.py --experiment $EXPERIMENT --expert $EXPERT --seed 0
```

Results are stored in the following folders:
```
experiments/
  toy4/
    plots/
    results/
```
