import torch
from torch import Tensor
from torch.nn import L1Loss


class FeatureMapLoss(torch.nn.L1Loss):
    def forward(
        self,
        feat_maps_true, feat_maps_pred,
        dis_name: str,
        **batch
    ) -> Tensor:
        loss = 0.
        for fm, fm_gen in zip(feat_maps_true, feat_maps_pred):
            loss += L1Loss(fm, fm_gen)
        return {f"feat_loss_{dis_name}": loss}
