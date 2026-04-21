# base image is pytorch with 11.8 CUDA and CUDNN8 (this combination worked well for me, feel free to change it)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

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

# Install Python dependencies
RUN pip install \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    h5py \
    open3d \
    opencv-python \
    tensorboard \
    einops

# Start as appuser
USER appuser