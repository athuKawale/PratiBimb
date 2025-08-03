# nvidia_drivers.sh
#— Add NVIDIA repos & install driver-550 (no change needed here) —#

sudo apt update
sudo apt install -y gnupg ca-certificates curl

# Pin the CUDA repo so it takes precedence
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Install NVIDIA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update

# Install the existing driver-550 you have
sudo ubuntu-drivers devices
sudo apt install -y nvidia-driver-550

sudo reboot