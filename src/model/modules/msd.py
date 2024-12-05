import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from torch.nn.utils import weight_norm, spectral_norm


class SubDiscriminator(nn.Module):
    def __init__(
        self,
        channels,
        kernels,
        strides,
        groups,
        normalizer=weight_norm,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        # channels = [1, 128, 128, 512, 1024, 1024, 1024]
        # kernels [15, 41, 41, 41, 41, 41, 5],
        # strides [1, 2, 2, 4, 4, 1, 1],
        # groups [1, 4, 16, 16, 16, 16, 1],
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    normalizer(
                        nn.Conv1d(
                            channels[i - 1],
                            channels[i],
                            kernels[i - 1],
                            strides[i - 1],
                            groups=groups[i - 1],
                            padding=(kernels[i - 1] - 1) // 2,
                        )
                    ),
                    nn.LeakyReLU(),
                )
                for i in range(1, len(channels))
            ]
        )

        self.conv_blocks.append(normalizer(nn.Conv2d(kernels[-1], 1, 3, 1, 1)))

    def forward(self, x):
        feat_maps = []
        for module in self.conv_blocks:
            x = module(x)
            feat_maps.append(x)

        x = torch.flatten(x, 1)
        return x, feat_maps


class MSD(nn.Module):
    def __init__(self, channels, kernels, strides, groups, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList(
            [
                SubDiscriminator(
                    channels, kernels, strides, groups, normalizer=spectral_norm
                ),
                SubDiscriminator(channels, kernels, strides, groups),
                SubDiscriminator(channels, kernels, strides, groups),
            ]
        )

        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x, x_gen) -> tp.Sequence[tp.List]:
        x_outs, x_gen_outs = [], []
        x_feat_maps, x_gen_feat_maps = [], []
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = self.pool(x)
            out, feat_map = block(x)
            x_outs.append(out)
            x_feat_maps.append(feat_map)
            out_gen, gen_feat_map = block(x_gen)
            x_gen_outs.append(out_gen)
            x_gen_feat_maps.append(gen_feat_map)

        return x_outs, x_gen_outs, x_feat_maps, x_gen_feat_maps
