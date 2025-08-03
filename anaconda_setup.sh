# ❇️ Step 0: Ensure system defaults aren’t broken
# (never remove Ubuntu’s python3 — it’s required by the OS)
sudo apt update

# Step 1: Install standard Ubuntu‑backed Python3 & pip
sudo apt install -y python3 python3-full python3-venv python3-pip python3-pip-whl
# (Add python‑is‑python3 if you want “python → python3” symlink)

# Verify:
python3 --version
python3 -m pip --version

# … Now optionally install Anaconda (Python 3.11.x base, ~4 GB)

# Step 2: Prepare tools and download installer
sudo apt install -y curl         # or use wget
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Step 3: (Strongly recommended) verify download
sha256sum Anaconda3-2024.02-1-Linux-x86_64.sh
# Compare with the official checksum listed on:
#  https://repo.anaconda.com/archive/  (look for matching hash)

# Step 4: Install silently into home directory
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p "$HOME/anaconda3"

# Step 5: Initialize and activate Conda
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

# Step 6: Add conda to startup
echo 'source "$HOME/anaconda3/etc/profile.d/conda.sh"' >> "$HOME/.bashrc"
echo 'conda activate --stack' >> "$HOME/.bashrc"

# (Optional) If using Conda environments for per-project isolation:
conda install -c conda-forge cudatoolkit cudnn nccl

# Restart shell (<no need to reboot>)
