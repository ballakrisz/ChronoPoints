# Acknowledgment

This folder contains an adapted version of the original implementation from the authors of *SequentialPointNet: Real-Time 3-D Human Action Recognition Based on Hyperpoint Sequence*.

The original codebase is available at:
https://github.com/XingLi1012/SequentialPointNet

We do not claim ownership of the original work. Minor modifications may have been made to accomodate the model four our own dataset.

If you use this model, please cite the original paper:
```
@ARTICLE{9954898,
  author={Li, Xing and Huang, Qian and Wang, Zhijian and Yang, Tianjin and Hou, Zhenjie and Miao, Zhuang},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Real-Time 3-D Human Action Recognition Based on Hyperpoint Sequence}, 
  year={2023},
  volume={19},
  number={8},
  pages={8933-8942},
  keywords={Point cloud compression;Three-dimensional displays;Task analysis;Computational modeling;Encoding;Solid modeling;Real-time systems;3-D action recognition;hyperpoint;point cloud sequence;SequentialPointNet},
  doi={10.1109/TII.2022.3223225}
}
```

## Installation

Build the Docker image by running:  

```bash
cd SequentialPointNet
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
python3 train.py --framenum 10 --seed 0
```
You can adjust the sequence length using the `framenum` parameter.

## Evaluation
To evaluate a trained model, run:
```bash
python3 test.py --framenum 10 --seed 0
```