import logging
from torch.utils.data import Dataset
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class CustomDirDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        # index_path: None | str
    ):
        self.dir_path = Path(data_dir)
        self._get_index()
        # self._index_path = index_path if index_path is not None else self.dir_path / "index.json"

    def _get_index(self):
        self._index = []
        wavs_dir = self.dir_path / "transcriptions"
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
        with open(path, 'r') as file:
            text = file.read()
            text = text.replace('\n', ' ')
            text = ' '.join(text.split())
            return text.strip()

        return text


    def __len__(self):
        return len(self._index)
