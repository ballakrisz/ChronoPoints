# base image is pytorch with 11.8 CUDA and CUDNN8 (this combination worked well for me, feel free to change it)
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

# Add NVIDIA CUDA GPG key (for some reason its missing in the base image)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update

# set shell to bash, set -c flag so the commands are interpreted as strings by deafult
SHELL ["/bin/bash", "-c"]

# set informative colors for the building process
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

# start with root user
USER root

# build-time argument given by build_docker.sh
ARG HOST_USER_GROUP_ARG

# create group appuser with id 999
# create grour hostgroup. This is needed so appuser can manipulate the host file without sudo
# create user appuser: home at /home/appuser, default shell is bash, id 999 and add to appuser group  
# set sudo password as admin for user appuser
# add user appuser to the following groups:
#   sudo (admin privis)
#   hostgroup
#   adm (system logging) --> might not be necessary
#   dip (network devices)
# finally, copy a .bashrc file into the container
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,adm,dip appuser && \
    cp /etc/skel/.bashrc /home/appuser/

# set working directory
WORKDIR /home/appuser

ENV DEBIAN_FRONTEND=noninteractive

# install basic dependencies for everything
USER root
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    jq \
    gdb \
    sudo \
    nano \
    net-tools \
    unzip \
    libgl1 \
    python3 python3-pip \
    cmake 

# install PointNet++ pytorch cuda ops
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN pip install ./Pointnet2_PyTorch/pointnet2_ops_lib

# Install Python dependencies
RUN pip install \
    numpy \
    tqdm \
    scikit-learn \
    matplotlib \
    optuna \
    optuna-dashboard \
    kaleido \
    pandas \
    plotly \
    optuna \
    optuna-dashboard \
    kaleido \
    einops \
    calflops



# PACKAGE INSTALL INSIDE CONTAINER:
    # sudo /opt/conda/bin/python -m pip install scikit-learn

# Installing their bullshit pytorch fork (cause why not reinvent the wheel)
# cd src
# mv modules-pytorch-1.8.1 modules
# cd modules
# mkdir -p pointnet2
# touch pointnet2/__init__.py
# python setup.py build_ext --inplace

# Start as appuser
USER appuser