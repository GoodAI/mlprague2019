########################################################################
# ML Prague 2019
# Workshop 
# High-precision classification of sound
########################################################################


##### Requirements for the workshop
# Linux instructions (Windows instructions are the same unless said so in the comments)

# Download and install anaconda for python 3.7
# download from https://www.anaconda.com/distribution/#download-section

# Run anaconda prompt, then create conda env with python 3.6 (because conda does not support tensorflow on 3.7):
conda create -n sound python=3.6.8

# Unset PYTHONPATH to avoid installation problems
export PYTHONPATH=  # On Windows, use: set PYTHONPATH=

# Change to the directory where you will work with the workshop code
cd sound_workshop

# Activate env
source activate sound  # On Windows, use: activate sound

# Install jupyter notebook if not present already
conda install jupyter

# Install conda core
conda install numpy pandas scikit-learn tqdm scipy -y

# Install pytorch
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# Install requirements
pip install pywavelets pycwt resampy eyed3 pydub soundfile
# additionally 'pip install python-magic-bin' on Windows

# Install tensorflow
conda install tensorflow

# Get VGGish data
wget https://storage.googleapis.com/audioset/vggish_model.ckpt  # on Windows, just download using browser
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz  # on Windows, just download using browser

# Get some sample wavs
# download from https://drive.google.com/file/d/1izMGO_2DquUu0RGdI4GfJm41UJDUJbBP/view
# we'll move these files to the appropriate folders during the workshop

# download a pre-trained ResNet model
# run python from command line, then:
from torch.utils.model_zoo import load_url
load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth")

# at the beginning of the workshop, you'll be provided with additional code (units of MB) that will be used during the workshop

# In anaconda prompt:
source activate sound  # On Windows, use: activate sound

# Clone the git repository (or copy from the provided flash drive).
git clone https://github.com/GoodAI/mlprague2019.git

cd mlprague2019

# Move the downloaded vggish*.ckpt and vggish*.npz to code/lib/vggish/
# Unzip all the wav files from wavs.zip into mlprague2019/wavs/

export PYTHONPATH=.  # On Windows, use: set PYTHONPATH=.

jupyter notebook

# Click on MLPrague Demo.ipynb
# In the third cell, change 'cuda' to 'cpu' if your machine doesn't have cuda capability
