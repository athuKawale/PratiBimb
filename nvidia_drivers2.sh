# nvidia_drivers2.sh
#— Install CUDA 12.0, cuDNN 9 (for CUDA 12), and NCCL 2.18 —#

# 1. Verify driver is running post-boot
nvidia-smi    # should still show Driver Version: 550

# 2. Install CUDA 12.0 via NVIDIA runfile (no driver overwrite)
CUDA_VER=12.0.1
CUDA_RUNFILE=cuda_${CUDA_VER}_525.85.12_linux.run
wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${CUDA_RUNFILE}
sudo sh ${CUDA_RUNFILE} --silent --toolkit --override

# 3. Set environment variables
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Install cuDNN 9 for CUDA 12 (v9.10.2.21)
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12

# 5. Install NCCL 2.18 (matches your libnccl2 2.18.5)
sudo apt install -y libnccl2=2.18.5-1-2 libnccl-dev=2.18.5-1-2

# 6. Validate setup
nvidia-smi
nvcc --version   # should report 12.0.x
dpkg -l | grep cudnn
dpkg -l | grep nccl

ln -sf $HOME/anaconda3/envs/pratibimb/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 $HOME/anaconda3/envs/pratibimb/lib/python3.11/site-packages/nvidia/cusparse/lib/libnvJitLink.so.12
