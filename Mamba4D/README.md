# Acknowledgment

This folder contains an adapted version of the original implementation from the authors of *Mamba4D: Efficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models*.

The original codebase is available at:
https://github.com/IRMVLab/Mamba4D

We do not claim ownership of the original work. Minor modifications may have been made to accomodate the model four our own dataset.

If you use this model, please cite the original paper:

```
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Jiuming and Han, Jinru and Liu, Lihao and Aviles-Rivero, Angelica I. and Jiang, Chaokang and Liu, Zhe and Wang, Hesheng},
    title     = {Mamba4D: Efficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17626-17636}
}
```

## Disclaimer

This repository is based on the original Mamba4D codebase, which depends on the `mamba-ssm` package (https://github.com/state-spaces/mamba). Recent updates to that package have dropped support for older GPUs. 

To ensure compatibility, this repository builds `mamba-ssm` from source, which is already included in the provided Dockerfile. However, to guarantee that PyTorch CUDA operations are compiled for your specific GPU architecture, you may need to review and adjust the **"Install mamba-ssm"** section in the Dockerfile accordingly.

## Installation

Build the Docker image by running:  
(due to having to build `mamba-ssm` from source, this step takes approximately 20-30 minutes)

```bash
cd Mamba4D
./build_docker.sh
```

After the build finishes, start the container and attach to it:
```bash
./run_docker.sh
```

The `run_docker.sh` script automatically runs `docker exec -it` after starting the container, placing you directly into an interactive terminal session inside the container.

## Training
To train the model on the provided dataset, run:
```bash
cd src
python3 train_mamba4d.py --clip-len 10 --seed 0
```
You can adjust the sequence length using the `clip-len` parameter.

## Evaluation
To evaluate a trained model, run:
```bash
python3 test_mamba4d.py --clip-len 10 --seed 0
```