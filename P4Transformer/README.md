# Acknowledgment

This folder contains an adapted version of the original implementation from the authors of *Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos*.

The original codebase is available at:
https://github.com/hehefan/P4Transformer

We do not claim ownership of the original work. Minor modifications may have been made to accomodate the model four our own dataset.

If you use this model, please cite the original paper:
```
@inproceedings{fan2021p4transformer,
    title     = {Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos},
    author    = {Fan, Hehe and Yang, Yi and Kankanhalli, Mohan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {14204--14213},
    doi       = {10.1109/CVPR46437.2021.01398}
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
python3 train_P4Transformer.py --clip-len 10 --seed 0
```
You can adjust the sequence length using the `clip-len` parameter.

## Evaluation
To evaluate a trained model, run:
```bash
python3 test_P4Transformer.py --clip-len 10 --seed 0
```