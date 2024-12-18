# HIFI GAN vocoder with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an implementation of HIFI gan for audio generation. This model converts mel spectrogram to `.wav` file. 

To generate audio file from text use text to spectrogram models. Like `Tacotron2` (see `txt2mel.py`). You can also generate audio from mel (audio -> mel -> audio) using `inference.py`.


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

Or download pretrained model weights:

```bash
python3 get_model_weights.py
```

To run inference audio -> mel -> audio (evaluate the model or save predictions. Uses `src/datasets/custom_dir_audio_dataset.py`, see `src/configs/inference.yaml`):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

To run inference text -> audio (evaluate the model or save predictions. Uses `src/datasets/custom_dir_dataset.py`, see `src/configs/inference_text.yaml`):

```bash
python3 txt2mel.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
