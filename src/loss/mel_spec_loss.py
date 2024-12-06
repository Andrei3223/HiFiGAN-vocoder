import torch
from torch import Tensor
from torch.nn import L1Loss


class MelSpecLoss(torch.nn.L1Loss):
    def forward(
        self, mel_input, mel_generated, **batch
    ) -> Tensor:
        loss = L1Loss()(mel_input, mel_generated)
        return {"mel_loss": loss}
