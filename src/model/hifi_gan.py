import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
# from src.model.modules.generator import Generator
from src.model.modules.mpd import MPD
from src.model.modules.msd import MSD

from src.model.delete import Generator


class HIFIGan(nn.Module):
    def __init__(self, generator, mpd, msd, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = Generator(**generator)
        self.mpd = MPD(**mpd)
        self.msd = MSD(**msd)

    def forward(self, mel):
        return self.generator(mel)
