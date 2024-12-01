import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from torch.nn.utils import weight_norm


class SubDiscriminator(nn.Module):
    def __init__(self, period, kernel=5, stride=3, *args, **kwargs) -> None:
        super().__init__()
        self._period = period

        channel_sizes = [1, 32, 128, 512, 1024]
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            channel_sizes[i - 1],
                            channel_sizes[i],
                            (kernel, 1),
                            (stride, 1),
                            ((kernel - 1) // 2, 0),
                        )
                    ),
                    nn.LeakyReLU(),
                )
                for i in range(1, len(channel_sizes))
            ]
        )
        self.conv_blocks.append(
            weight_norm(nn.Conv2d(1024, 1024, (kernel, 1), 1, (2, 0)))
        )
        self.conv_blocks.append(weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0))))

    def forward(self, x):
        feat_maps = []
        bs, ch, t_len = x.shape

        if t_len % self._period != 0:
            remainder = self._period - (t_len % self.period)
            x = F.pad(x, (0, remainder), "reflect")
        x = x.reshape(bs, ch, t_len // self.period, self.period)

        for module in self.convs:
            x = module(x)
            feat_maps.append(x)
        x = torch.flatten(x, 1)
        return x, feat_maps


class MPD(nn.Module):
    def __init__(self,
                 periods,
                 kernel,
                 stride,
                 channels):
        super().__init__()
        self.blocks = nn.ModuleList([
                            SubDiscriminator(
                                p,
                                kernel,
                                stride,
                                channels
                            )
                            for p in periods
                        ])

    def forward(self, x, x_gen) -> tp.Sequence[tp.List]:
        x_outs, x_gen_outs = [], []
        x_feat_maps, x_gen_feat_maps = [], []
        for block in self.blocks:
            out, feat_map = block(x)
            x_outs.append(out)
            x_feat_maps.append(feat_map)
            out_gen, gen_feat_map = block(x_gen)
            x_gen_outs.append(out_gen)
            x_gen_feat_maps.append(gen_feat_map)

        return x_outs, x_gen_outs, x_feat_maps, x_gen_feat_maps
