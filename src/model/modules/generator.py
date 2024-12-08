import torch.nn as nn
import typing as tp
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, hidden, kernel, Dilations, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList(
            list(
                nn.Sequential(
                    nn.LeakyReLU(),
                    # TODO nn.utils.weight_norm ???
                    nn.Conv1d(
                        in_channels=hidden,
                        out_channels=hidden,
                        kernel_size=kernel,
                        dilation=Dilations[i][j],
                        padding="same",
                    ),
                )
                for i in range(len(Dilations))
                for j in range(len(Dilations[i]))
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MRF(nn.Module):
    def __init__(
        self,
        hidden: int,
        kernels: tp.List[int],
        dilations: tp.List[tp.List[tp.List[int]]],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, kernel, dilations[i]) for i, kernel in enumerate(kernels)]
        )

    def forward(self, x):
        out = self.blocks[0](x)
        for block in self.blocks[1:]:
            out += block(x)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        kernels_u: tp.List[int],
        kernels_mrf: tp.List[int],
        dilations: tp.List[tp.List[tp.List[int]]],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_block = nn.Conv1d(
                channels, hidden, kernel_size=7, dilation=1, padding="same"
            )


        self.mrf_blocks = nn.Sequential(
            *list(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(
                        hidden // (2**i),
                        hidden // (2 ** (i + 1)),
                        kernel_size=kernel_u,
                        stride=kernel_u // 2,
                        padding=kernel_u // 4,
                    ),
                    MRF(hidden // (2 ** (i + 1)), kernels_mrf, dilations),
                )
                for i, kernel_u in enumerate(kernels_u)
            )
        )
        self.out_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=hidden // (2 ** len(kernels_u)),
                out_channels=1,
                kernel_size=7,
                padding="same",
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.in_block(x)
        out = self.mrf_blocks(out)
        out = self.out_block(out)
        return out
