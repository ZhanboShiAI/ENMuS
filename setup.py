from setuptools import setup

setup(
    name='sen',
    version="1.0.0",
    packages=[
        "sen",
        "sen_baselines"
    ],
    install_requires=[
        'torch',
        'gym',
        'numpy>=1.16.1',
        'yacs>=0.1.5',
        'numpy-quaternion>=2019.3.18.14.33.20',
        'attrs>=19.1.0',
        'opencv-python>=3.3.0',
        'imageio>=2.2.0',
        'imageio-ffmpeg>=0.2.0',
        'scipy>=1.0.0',
        'tqdm>=4.0.0',
        'numba',
        'Pillow',
        'pydub',
        'getch',
        'matplotlib',
        'librosa',
        'torchsummary',
        'gitpython',
        'networkx',
        'tqdm',
        'notebook',
        'moviepy',
        'tensorflow',
        'astropy',
        'scikit-image'
    ],
)