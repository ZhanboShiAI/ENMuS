<p align="center">
  <h1 align="center">
    Towards Audio-visual Navigation in Noisy Environments: A Large-scale Benchmark Dataset and An Architecture Considering Multiple Sound-Sources
   <br>
    [AAAI 2025]
  </h1>

  <p align="center">
  <a href="https://github.com/ZhanboShiAI"><strong>Zhanbo Shi</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=8VOk_S4AAAAJ&hl=en"><strong>Lin Zhang*</strong></a>
  ·
  <a href="https://github.com/lif314"><strong>Linfei Li</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=A0N_mS0AAAAJ&hl=en"><strong>Ying Shen</strong></a>
</p>

## Motivation
Audio-visual navigation has received considerable attention in recent years. However, the majority of related investigations have focused on single sound-source scenarios. Studies in this field for multiple sound-source scenarios remain underexplored due to the limitations of two aspects. First, the existing audio-visual navigation dataset only has limited audio samples, making it difficult to simulate diverse multiple sound-source environments. Second, existing navigation frameworks are mainly designed for single sound-source scenarios, thus their performance is severely reduced in multiple sound-source scenarios. To fill in these two research gaps to some extent, we establish a large-scale audio dataset named **BeDAViN** and propose a new embodied navigation framework called **ENMuS<sup>3</sup>**.

## Citation
If you use this project in your research, please cite the following [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33608):
```
@inproceedings{shi2025towards,
  title={Towards Audio-Visual Navigation in Noisy Environments: A Large-Scale Benchmark Dataset and an Architecture Considering Multiple Sound-Sources},
  author={Shi, Zhanbo and Zhang, Lin and Li, Linfei and Shen, Ying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={14},
  pages={14673--14680},
  year={2025}
}
```

## Appendix
The appendix of our paper can be found [here](./Appendix.pdf).

## Environment Installation
This project is developed with Python 3.9 on Ubuntu 22.04. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment with following instructions. 

```bash
conda create -n enmus python=3.9 cmake=3.14.0 -y
conda activate enmus
```

ENMuS<sup>3</sup> uses [Habitat](https://github.com/facebookresearch/habitat-lab) and [SoundSpaces](https://github.com/facebookresearch/sound-spaces) for robot simulation and audio synthesis.

First, install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.2) v0.2.2 and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.2) v 0.2.2 with the following commands.

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
```

Edit habitat/tasks/rearrange/rearrange_sim.py file and remove the 36th line where FetchRobot is imported.

Then install [SoundSpaces](https://github.com/facebookresearch/sound-spaces) as the instructions [here](https://github.com/facebookresearch/sound-spaces/blob/main/INSTALLATION.md).

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

Now you can install ENMuS<sup>3</sup>:

```bash
git clone https://github.com/ZhanboShiAI/ENMuS.git
cd ENMuS
pip install -e .
```

## Data
To use ENMuS<sup>3</sup>, you should firstly generate the following data. 

**Scene Dataset from Matterport3D**

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). After the scene dataset has been downloaded, you can use the `cache_observations.py` script from SoundSpaces to generate cached observations. 

**Room Impulse Response Dataset from SoundSpaces**

Room impulse response dataset is used for audio synthesis. You can download this dataset following the instructions [here](https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md).

**Audio Dataset**

The Audio dataset used in this project from three parts, our manual recorded audio files, audio samples selected from [AudioSet](https://research.google.com/audioset/) and [Freesound](https://annotator.freesound.org/). The processed audio wavfiles are saved under `data` directory of the current github repository. The raw 24 bit audio recordings with a sampling rate of 96,000 Hz can be found [here](https://zenodo.org/records/14486235). As for the information of the original audio clips from AudioSet and Freesound, please refer to [this document](./data/sounds/metadata/README.md). 

**Data Folder Structure**
```
data|
    ├── binaural_rirs                             # binaural RIRs of 2 channels
    │   └── [dataset]
    │       └── [scene]
    │           └── [angle]                       # azimuth angle of agent's heading in mesh coordinates
    │               └── [receiver]-[source].wav
    ├── datasets                                  # stores datasets of episodes of different splits
    │   └── [dataset]
    │       └── [version]
    │           └── [split]
    │               ├── [split].json.gz
    │               └── content
    │                   └── [scene].json.gz
    ├── metadata                                  # stores metadata of environments
    │   └── [dataset]
    │       └── [scene]
    │           ├── point.txt                     # coordinates of all points in mesh coordinates
    │           ├── graph.pkl                     # points are pruned to a connectivity graph
    ├── pretrained_weights                        # saved pretrained weights for the model
    │   └── [semantic_audionav]
    │       └── [enmus]
    │           ├── enmus_best_val.pth
    ├── scene_datasets                            # scene_datasets
    │   └── [dataset]
    │       └── [scene]
    │           └── [scene].house (habitat/mesh_sementic.glb)
    └── scene_observations                        # pre-rendered scene observations
    │   └── [dataset]
    │       └── [scene].pkl                       # dictionary is in the format of {(receiver, rotation): sim_obs}
    ├── sounds                                    # stores all sounds
    │   └── sound_event_splits
    │       ├── noise                             # stores sounds for background noise simulation
    │       ├── sound_event                       # stores sounds for sound event simulation
    |           ├── test                          # splits
    |           ├── train
    |           ├── val
    |               └── [sound_id].wav
```

## Usage
Below we show some example commands for training and evaluating ENMuS<sup>3</sup> on Matterport3D in multi-source scenarios.
1. Training
```
python sen_baselines/enmus/run.py --exp-config sen_baselines/enmus/config/multi_source/enmus.yaml --decoder-type MSMT --model-dir data/models/mp3d/enmus_multi_source
```

2. Validation (evaluate every 10 checkpoints and generate a validation curve)
```
python sen_baselines/enmus/run.py --exp-config sen_baselines/enmus/config/multi_source/enmus_eval.yaml --run-type eval --decoder-type MSMT --model-dir data/models/mp3d/enmus_multi_source --eval-interval 10
```

3. Test the best validation checkpoint based on validation curve
```
python sen_baselines/enmus/run.py --exp-config sen_baselines/enmus/config/multi_source/enmus_test.yaml --run-type eval --decoder-type MSMT --model-dir data/models/mp3d/enmus_multi_source EVAL_CKPT_PATH_DIR data/models/mp3d/enmus_multi_source/data/ckpt.XXX.pth
```

## License
The codebase of this project and our manually collected audio recordings are CC-BY-4.0 licensed, as found in the [LICENSE](./LICENSE) file. The audio samples from AudioSet and Freesound are mainly under CC-BY-4.0 license. A detailed license info of these audio samples can be found under `data/sounds/metadata/`. 

The trained RL policy models and task datasets are considered data derived from the MP3D scene dataset. Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
