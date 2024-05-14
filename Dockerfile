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
RUN apt update && apt install -y git vim libgl1-mesa-glx
RUN apt-get install -y wget libglib2.0-0

# Install Selfharm Project
RUN git clone https://github.com/qqaazz0222/PoliceLab_Plass_Selfharm

# Setting Selfharm Project
RUN mv ./PoliceLab_Plass_Selfharm/* ./
RUN rm -rf ./PoliceLab_Plass_Selfharm/

# Install BoTSORT
RUN git clone https://github.com/qqaazz0222/PoliceLab_Plass_BoTSORT

# Setting BoTSORT
RUN mv ./PoliceLab_Plass_BoTSORT ./BoTSORT

# Install Library
RUN pip install -r ./PLASS/requirements.txt
RUN pip install -r ./BoTSORT/requirements.txt

# Download Weight Filse (Selfharm)
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EepW2QK6ygJMpTSpk7zsiXoBo3NSq4KibNp-tAgOsCNRDA?e=EqOkKb&download=1" -O PLASS/model/selfharm_pose_checkpoint_best.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EaKk7gy9dq5AgyaBNK7gVcYByDjnz2mK7eQ0wyrPYdstug?e=1latwR&download=1" -O PLASS/model/human_pose_checkpoint.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ebk9j6XmHZNKkjTB95HXLOgBK856p99nlb2jMOeuKkeUYg?e=H3znvQ&download=1" -O PLASS/model/human_detect_checkpoint.pth
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZypUOJgyl5MnlnaBm0B5WMBMmmf-5hhh4-8xmOJ-vgJAQ?e=2Fl0MU&download=1" -O PLASS/model/human_detect_tracker_checkpoint.pt
# Download Weight Filse (BoTSORT)
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EVRucL6WrilAvrvYqVLI0sUBDJPJ7xp9fMqfPdkf1X66UA?e=lj2jEE&download=1" -O BoTSORT/3_epoch_099.pt
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EYKSiqe1zMRDhMhyFEbwy0cBDkpGgPoS4BF91AHJjOsSkw?e=7WunKf&download=1" -O BoTSORT/6_epoch_049.pt
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZl03KMZENRPuRIU8KJfp-wBBjFCaLEcpLEFpU95YF8x5w?e=YbEHwZ&download=1" -O BoTSORT/best.pt
RUN wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EVarSDVpb-dIl2dqjJvnXgoBHfNc12DlvrddyT1ASr-qTQ?e=GJxKkF&download=1" -O BoTSORT/yolov7_custom.pt

# Set the default command to run when the container starts
CMD ["bash"] 