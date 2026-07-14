# base image is pytorch with 11.8 CUDA and CUDNN8 (this combination worked well for me, feel free to change it)
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN apt-get update

# set shell to bash, set -c flag so the commands are interpreted as strings by deafult
SHELL ["/bin/bash", "-c"]

# start with root user
USER root

# set informative colors for the building process
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

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

# Avoid interactive prompts
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
    tqdm \
    pyyaml \
    einops \
    scikit-learn \
    matplotlib \
    opencv-python \
    open3d \
    timm \
    termcolor \
    calflops

# # install knn CUDA
# RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl && \
#     chown -R appuser:appuser /opt/conda/lib/python3.11/site-packages/knn_cuda

# install PointNet++ pytorch cuda ops
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN pip install ./Pointnet2_PyTorch/pointnet2_ops_lib

# Ensure correct CUDA arch (its probably not needed as we hard wire our arch into the setup.py file, because this alone was not enough)
ENV TORCH_CUDA_ARCH_LIST=6.1

# Install mamba-ssm
# Make sure to build against the Compute Capability (CC) of YOUR GPU. 
    # You can either get your CC by looking up your GPU model at: 
        # https://developer.nvidia.com/cuda/gpus

    # Or you can get it via PyTorch: 
        # python3 -c "import torch; print(torch.cuda.get_device_capability())"

# Once you have it, replace this line:
    # print indent "cc_flag.append(\"arch=compute_61,code=sm_61\")";\
# If your CC is 10.3, then your line should look like this:
    # print indent "cc_flag.append(\"arch=compute_103,code=sm_103\")";\

RUN git clone --branch v1.2.0 --depth 1 https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    awk '{\
        if (!done && $0 ~ /cc_flag.append\("-gencode"\)/) {\
            indent = substr($0, 1, match($0, /[^ ]/) - 1);\
            print indent "cc_flag.append(\"-gencode\")";\
            print indent "cc_flag.append(\"arch=compute_61,code=sm_61\")";\
            done=1;\
        }\
        print $0;\
    }' setup.py > setup_tmp.py && mv setup_tmp.py setup.py && cat setup.py && \
    MAMBA_FORCE_BUILD=TRUE pip install . --no-build-isolation && \
    cd .. && rm -rf mamba

# Start as appuser
USER appuser