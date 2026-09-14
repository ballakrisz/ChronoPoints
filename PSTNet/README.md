# Acknowledgment

This folder contains an adapted version of the original implementation from the authors of *PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences*.

The original codebase is available at:
https://github.com/hehefan/Point-Spatio-Temporal-Convolution

We do not claim ownership of the original work. Minor modifications may have been made to accomodate the model four our own dataset.

If you use this model, please cite the original paper:
```
@inproceedings{fan2021pstnet,
  author    = {Haokui Fan and Hao Dong and Yuchao Dai and Jianwei Guo and Zhen Dong and Junchen Ye and Mingyi He},
  title     = {PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences},
  booktitle = {Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2021},
  pages     = {1277--1285},
}
```

## Installation

Build the Docker image by running:  

```bash
cd PSTNet
./build_docker.sh
```

After the build finishes, start the container and attach to it:
```bash
./run_docker.sh
```

The `run_docker.sh` script automatically runs `docker exec -it` after starting the container, placing you directly into an interactive terminal session inside the container.


## Hyperparameter tuning
To optimize the hyperparameters of the model run:
```bash
cd src
python3 optuna_tuning.py
```

## Training
To train the model on the provided dataset, run:
```bash
cd src
python3 train_PSTNet.py --clip-len 10 --seed 0
```
You can adjust the sequence length using the `clip-len` parameter.

## Evaluation
To evaluate a trained model, run:
```bash
python3 test_PSTNet.py --clip-len 10 --seed 0
```