from pathlib import Path

import logging
from torch.utils.data import Dataset
from pathlib import Path
import os
from src.transforms.mel_spec import MelSpectrogram
import torchaudio

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        target_sr: int,
        *args, **kwargs
    ):
        self.dir_path = Path(data_dir)
        self.target_sr = target_sr
        self._get_index()
        self.get_mel_spec = MelSpectrogram()

    def _get_index(self):
        self._index = []
        wavs_dir = self.dir_path
        for root, _, files in os.walk(wavs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._index.append(
                    {
                        "path": file_path,
                    }
                )

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        path = data_dict["path"]
        audio = self.load_audio(path)
        spectrogram = self.get_mel_spec(audio)
        return spectrogram

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def __len__(self):
        return len(self._index)

