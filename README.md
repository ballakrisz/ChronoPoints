# Title
This repository contains the code for the paper [].

## Dataset
Download the dataset from:  
https://drive.google.com/file/d/1Xfz22RLZl0_2RmsJ2Wj2Fg8v-ongBgAE/view?usp=sharing

Extract it so that your folder structure looks like this:
```bash
dataset/
└── chrono_points_cls_benchmark/
```

## Benchmark Models, Attribution, and Usage
This repository includes implementations of several third-party models for benchmarking purposes. All original implementations belong to their respective authors, and links to the official repositories along with citation instructions are provided in each model's folder. We only provide adapted versions to enable consistent training and evaluation on our dataset.

As the models require different environments, a separate `Dockerfile` is provided for each of them. **To train and evaluate both the comparison models and our own model, please refer to the README files in the respective folders.**

## Dependencies

The project has been tested on **Ubuntu 22.04** with the following GPUs:
- NVIDIA GeForce GTX 1080 Ti  
- NVIDIA GeForce RTX 3080 Ti  

(Though you'll have to manually compile custom TensorFlow GPU ops against newer video cards to use MeteorNet. For the 1080Ti  we provide them already compiled.)
### Docker

Docker is required to run the project. Install it by following the official guide:

👉 https://docs.docker.com/engine/install/ubuntu/

Use the **APT repository installation method**.

After installation, run the following commands to avoid having to use `sudo` with the docker cli:

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```
This adds your user to the `docker` group, allowing you to execute Docker commands without elevated privileges.

### NVIDIA Container Toolkit (GPU Support)
To enable GPU acceleration inside Docker containers, install the NVIDIA Container Toolkit:

👉 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Follow the **APT repository setup** instructions, and be sure to complete the **“Configuring Docker”** section to properly enable GPU support.