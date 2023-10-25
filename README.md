# Active-Negative-Loss-Functions

## Requirements

```console
python >= 3.9, torch >= 1.12.1, torchvision >= 0.13.1, numpy >= 1.23.1
```

## How to use

### Configs

Check '*.json' file in the config folder for each exeriment.

### Arguments

* gpu: GPU id
* seed: random seed
* config: config name
* noise_type: 'sym' if use symmetric noise, 'asym' if use asymmetric noise
* noise_rate: noise rate
* eval_freq: frequency of evaluation, default is 1
* tuning: use the tuning settings (90% of the original training set as training set and 10% as validation set)

### Example

Training ANL-CE on CIFAR-10 with 0.8 symmetric noise:
```bash
python main.py \
--gpu 0 \
--seed 1 \
--config cifar10_anl_ce \
--noise_type sym \
--noise_rate 0.8 \
--eval_freq 10
```

## Thanks

Moreover, we thank the codes implemented by [Ma et al.](https://github.com/HanxunH/Active-Passive-Losses) and [Zhou et al.](https://github.com/hitcszx/ALFs).