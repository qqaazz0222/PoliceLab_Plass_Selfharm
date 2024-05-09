ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y gnupg2
RUN apt update && apt install -y git vim libglib2.0-0 libgl1-mesa-glx

WORKDIR /selfharm_PLASS

RUN git clone https://github.com/qqaazz0222/PoliceLab_Plass_Selfharm.git .

RUN cd PoliceLab_Plass_Selfharm

RUN pip install -e .

RUN apt-get update && apt-get install -y wget

RUN bash donwload_checkpoint.sh

CMD ["bash"]