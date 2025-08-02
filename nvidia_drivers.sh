# Step 1: Add NVIDIA Package Repositories and Keys (latest CUDA/driver support, Ubuntu 24.04+)

sudo apt install -y gnupg ca-certificates curl

# (Recommended) Add CUDA apt priority pin, so CUDA repo takes precedence
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Install the NVIDIA-maintained keyring package (fetches all correct repo keys & sources)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update APT cache
sudo apt update

# Step 2: Install recommended or specific NVIDIA driver
sudo ubuntu-drivers devices                    # Shows recommended version
sudo apt install -y nvidia-driver-550          # Or use recommended version (update this if needed)

# Step 3: Reboot to activate driver
sudo reboot