<p align="center">
  <h1 align="center">
    Environment Installation
  </h1>

## Versions
PyTorch version: 2.4.0+cu124  
Is debug build: False  
CUDA used to build PyTorch: 12.4  
ROCM used to build PyTorch: N/A  

OS: Ubuntu 22.04.5 LTS (x86_64)  
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0  
Clang version: Could not collect  
CMake version: version 3.14.0  
Libc version: glibc-2.35  

Python version: 3.9.23 (main, Jun  5 2025, 13:40:20)  [GCC 11.2.0] (64-bit runtime)  
Python platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.35  
Is CUDA available: True  
CUDA runtime version: 12.4.99  
CUDA_MODULE_LOADING set to: LAZY  
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB  
Nvidia driver version: 570.133.07  
cuDNN version: Could not collect  
HIP runtime version: N/A  
MIOpen runtime version: N/A  
Is XNNPACK available: True  

Versions of relevant libraries:  
[pip3] numpy==1.26.4  
[pip3] numpy-quaternion==2023.0.4  
[pip3] optree==0.16.0  
[pip3] torch==2.4.0+cu124  
[pip3] torchaudio==2.4.0+cu124  
[pip3] torchsummary==1.5.1  
[pip3] torchvision==0.19.0+cu124  
[pip3] triton==3.0.0  

## Installation
Create conda environment

```bash
conda create -n enmus python=3.9 cmake=3.14.0 -y
conda activate enmus
```

Install pytorch 2.4.0

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

Install habitat-sim v0.2.2

```bash
sudo apt-get update || True
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev

git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 80f8e31140eaf50fe6c5ab488525ae1bdf250bd9
pip install -r requirements.txt
python setup.py install --with-cuda --audio --headless --bullet
```

Install habitat-lab v0.2.2

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout 0f454f62e41050bc90ca468c62db35d7484923ff
pip install -r requirements.txt
python setup.py develop --all
```

Install Sound-spaces

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

Install ENMuS
```bash
git clone https://github.com/ZhanboShiAI/ENMuS.git
cd ENMuS
pip install -e .
```

## Additional Notes
Due to the confilicts between pytorch and tensorflow, if you import tensorflow before pytorch, you may encounter the `RuntimeError: random_device could not be read` error. To solve it, we import pytorch before tensorflow in line 8-9 in `sen_baselines/enmus/run.py`. 

If you want to install the environment with CUDA 11, you can follow the instructions in this [link](https://github.com/facebookresearch/sound-spaces/issues/80#issuecomment-1195121562). 