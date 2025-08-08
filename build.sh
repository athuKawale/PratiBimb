mkdir models
git clone https://huggingface.co/AthuKawaleLogituit/Faceswap
rm -rf Faceswap/.git
mv Faceswap/* ./models/
rm -rf Faceswap
pip install -r requirements.txt
conda run -n pratibimb --live-stream conda install -c conda-forge libstdcxx-ng -y
echo 'export NO_ALBUMENTATIONS_UPDATE="1"' >> ~/.bashrc
source ~/.bashrc
echo $NO_ALBUMENTATIONS_UPDATE