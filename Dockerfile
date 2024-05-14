ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

# Install Torch
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Setting Working Directory
WORKDIR /SelfharmDetetion_PLASS

# Install Package
RUN apt update && apt install -y git vim libgl1-mesa-glx wget

# Install Project
RUN git clone https://github.com/qqaazz0222/PoliceLab_Plass_Selfharm

# Directory Fix
RUN mv PoliceLab_Plass_Selfharm/ ./
RUN rm -rf PoliceLab_Plass_Selfharm/

# Download Weight Filse
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EepW2QK6ygJMpTSpk7zsiXoBo3NSq4KibNp-tAgOsCNRDA?e=EqOkKb&download=1" -O PLASS/model/selfharm_pose_checkpoint_best.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EaKk7gy9dq5AgyaBNK7gVcYByDjnz2mK7eQ0wyrPYdstug?e=1latwR&download=1" -O PLASS/model/human_pose_checkpoint.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ebk9j6XmHZNKkjTB95HXLOgBK856p99nlb2jMOeuKkeUYg?e=H3znvQ&download=1" -O PLASS/model/human_detect_checkpoint.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZypUOJgyl5MnlnaBm0B5WMBMmmf-5hhh4-8xmOJ-vgJAQ?e=2Fl0MU&download=1" -O PLASS/model/human_detect_tracker_checkpoint.pt

# Set the default command to run when the container starts
CMD ["bash"] 