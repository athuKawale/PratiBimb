conda create -n pratibimb python=3.11 -y
conda activate pratibimb
mkdir models
git clone https://huggingface.co/AthuKawaleLogituit/Faceswap
rm -rf Faceswap/.git
mv Faceswap/* ./models/
rm -rf Faceswap
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng