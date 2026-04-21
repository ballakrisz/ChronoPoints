# Acknowledgment

This folder contains an adapted version of the original implementation from the authors of *MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences*.

The original codebase is available at:
https://github.com/xingyul/meteornet

We do not claim ownership of the original work. Minor modifications may have been made to accomodate the model four our own dataset.

If you use this model, please cite the original paper:
```
@inproceedings{liu2019meteornet, 
  title={MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences}, 
  author={Xingyu Liu and Mengyuan Yan and Jeannette Bohg}, 
  booktitle={ICCV}, 
  year={2019} 
}
```

## Disclaimer

This repository builds upon the original MeteorNet codebase, which includes custom TensorFlow GPU operations. For your convenience, we provide precompiled versions of these operations using CUDA 9 to replicate the original authors' environment.

**Note:** If your GPU uses a newer architecture that is incompatible with CUDA 9.x (e.g., recent NVIDIA GPUs), you will need to manually compile the custom operations. Please refer to the original repository for build instructions in that case.

## Installation

Build the Docker image by running:

```bash
cd MeteorNet
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
./command_train.sh
```
You can adjust the sequence length using the `num_frame` parameter in the shell script.

## Evaluation
To evaluate a trained model, run:
```bash
./command_test.sh
```