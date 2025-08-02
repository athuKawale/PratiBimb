# After reboot, verify driver installation:
nvidia-smi

# Step 1: Install CUDA Toolkit (latest from NVIDIA's repo)
sudo apt update
sudo apt install -y cuda

# (Optional) Install specific CUDA version if required, e.g.:
# sudo apt install -y cuda-toolkit-12-8

# Step 2: Add CUDA environment variables permanently
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA compiler version
nvcc --version

# Step 3: Install cuDNN libraries (from NVIDIA's repo)
sudo apt install -y libcudnn8 libcudnn8-dev

# Step 4: Install NCCL libraries
sudo apt install -y libnccl2 libnccl-dev

# Step 5: Validate setup:
nvidia-smi                       # Confirm GPU & driver
nvcc --version                   # Confirm CUDA
dpkg -l | grep nccl              # Confirm NCCL

# (Optional) Validate with PyTorch in Python:
# python -c "import torch; print(torch.cuda.is_available())"
# python -c "import torch; print(torch.backends.cudnn.enabled)"
# python -c "import torch; print(torch.distributed.is_nccl_available())"
