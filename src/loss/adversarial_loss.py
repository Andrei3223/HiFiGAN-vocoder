import torch
from torch import Tensor


class DiscriminatorAdvLoss(torch.nn.L1Loss):
    def forward(d_x_list, d_g_list, name: str, **batch):
        loss = 0.
        for d_x, d_g in zip(d_x_list, d_g_list):
            loss += torch.mean((d_x - 1) ** 2) + torch.mean(d_g ** 2)
        return {f"discriminator_loss_{name}": loss}

class GeneratorAdvLoss(torch.nn.L1Loss):
    def forward(x_list, name: str, **batch):
        loss = 0.
        for x in x_list:
            loss += torch.mean((x - 1) ** 2)
        return {f"generator_loss_{name}": loss}
