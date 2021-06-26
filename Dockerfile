# Start the basic stack
# FROM ubuntu:16.04

# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt update -y
RUN apt upgrade -y

# install project dependencies
RUN apt-get update
RUN apt-get update \
        && apt-get install --no-install-suggests -y \
        		vim \
                git \
                nano \
                python3 \
                python3-pip \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
        
# install pytorch
RUN pip3 install --upgrade pip \
        && pip3 install torch==1.6.0 torchvision==0.7.0 \
        && pip3 install --upgrade matplotlib

RUN pip install Pillow \
        && pip install tqdm \
        && pip install tensorboard \
        && pip install jupyterlab \
	&& pip install tensorboardX
RUN pip install scikit-image

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN pip install opencv-python
RUN pip install nuscenes-devkit

CMD /bin/bash
