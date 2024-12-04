import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from src.model.modules.generator import Generator
from src.model.modules.mpd import MPD
from src.model.modules.msd import MSD


class HIFIGan(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = Generator(config["generator"])
        self.mpd = MPD(config["mpd"])
        self.msd = MSD(config["msd"])

    def forward(self, mel):
        return self.generator(mel)
