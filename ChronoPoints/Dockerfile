# base image is pytorch with 11.8 CUDA and CUDNN8 (this combination worked well for me)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# common practice
RUN apt-get update

# set shell to bash, set -c flag so the commands are interpreted as strings by deafult
SHELL ["/bin/bash", "-c"]

# set informative colors for the building process
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

# start with root user
USER root

# build-time argument given by build_docker.sh
ARG HOST_USER_GROUP_ARG
ARG USE_VSCODE

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

# install basic dependencies for everything 
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noniteractive \
    apt-get install -y \
    git \
    git-lfs \
    build-essential \
    wget \
    curl \
    jq \
    gdb \
    sudo \
    nano \
    net-tools \
    unzip \
    cmake \
    python3-dev \
    chromium-browser \
    python3-tk

# Install Python dependencies
RUN pip install \
    scikit-learn \
    matplotlib \
    tensorboard \
    calflops \
    transformers \
    pandas \
    optuna \
    optuna-dashboard \
    kaleido \
    plotly \
    torchinfo

# install PointNet++ pytorch cuda ops (needed for FPS sampling and sptatiotemporal grouping)
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN pip install ./Pointnet2_PyTorch/pointnet2_ops_lib

# add tensorboard to PATH so you can run it with: tensorboard --logidr [logdir] --port [6006 or whatever you want]
ENV PATH="/home/appuser/.local/bin:${PATH}"

# start as appuser
USER appuser