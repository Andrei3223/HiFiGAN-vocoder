import json
import os
import shutil
from pathlib import Path

import torchaudio

# import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from src.utils.init_utils import set_random_seed

import requests
import os
import tarfile
from sklearn.model_selection import train_test_split
import numpy as np


def download_ljspeech():
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    filename = "LJSpeech-1.1.tar.bz2"

    # Download
    print("Downloading LJSpeech dataset")
    response = requests.get(url, verify=False)
    with open(filename, "wb") as f:
        f.write(response.content)

    # Extract
    print("Extracting files")
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall()

    # Clean up
    os.remove(filename)
    print("Download and extraction complete!")


URL_LINK = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJSpeechDataset(BaseDataset):
    def __init__(
        self, part, data_dir=None, seed=42, val_size=0.2, *args, **kwargs
    ) -> None:
        self._seed = seed
        self._val_size = val_size
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._part = part
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _load(self):
        print("load dataset")
        arch_path = os.path.join(self._data_dir, f"lj.tar.bz2")
        print("Downloading LJSpeech dataset")
        response = requests.get(URL_LINK, verify=False)
        with open(arch_path, "wb") as f:
            f.write(response.content)

        print("Extracting files")
        with tarfile.open(arch_path, "r:bz2") as tar:
            tar.extractall(path=self._data_dir)

        os.remove(arch_path)
        print("Download and extraction complete!")

    def _get_or_load_index(self, part):
        index_path = Path(self._data_dir) / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
            return index

        print("Creating indexes")
        train_path = Path(self._data_dir) / f"train_index.json"
        val_path = Path(self._data_dir) / f"val_index.json"
        train_index, val_index = self._create_index()
        with train_path.open("w") as f:
            json.dump(train_index, f, indent=2)
        with val_path.open("w") as f:
            json.dump(val_index, f, indent=2)

        return self._get_or_load_index(part)

    def _create_index(self):
        set_random_seed(self._seed)
        index = []
        data_dir = Path(self._data_dir) / "LJSpeech-1.1"
        wavs_dir = data_dir / "wavs"
        # print(self._data_dir.exists(), self._data_dir)
        if not data_dir.exists():
            self._load()
        else:
            print("Repo exists")

        for root, _, files in os.walk(wavs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                t_info = torchaudio.info(file_path)
                index.append(
                    {
                        "path": file_path,
                        "audio_len": t_info.num_frames / t_info.sample_rate,
                    }
                )

        train_idx, val_idx = train_test_split(
            np.arange(len(index)), test_size=self._val_size, random_state=self._seed
        )
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        index = np.array(index)
        return index[train_idx].tolist(), index[val_idx].tolist()
