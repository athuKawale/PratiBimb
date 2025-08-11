git clone --single-branch --branch jagrut https://github.com/athuKawale/PratiBimb.git
cd PratiBimb
mkdir models
git clone https://huggingface.co/AthuKawaleLogituit/Faceswap
rm -rf Faceswap/.git
mv Faceswap/* ./models/
rm -rf Faceswap
pip install -r requirements.txt
conda run -n pratibimb --live-stream conda install -c conda-forge libstdcxx-ng -y
echo 'export PYTHONPATH="$(pwd)"' >> ~/.bashrc
echo 'export NO_ALBUMENTATIONS_UPDATE="1"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/anaconda3/envs/pratibimb/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONWARNINGS="ignore:resource_tracker:UserWarning"' >> ~/.bashrc
source ~/.bashrc