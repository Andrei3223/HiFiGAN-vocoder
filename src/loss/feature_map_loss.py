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
        loss_func = L1Loss()
        for fm_list, fm_gen_list in zip(feat_maps_true, feat_maps_pred):
            for fm, fm_gen in zip(fm_list, fm_gen_list):
                loss += loss_func(fm, fm_gen)
        return {f"feat_loss_{dis_name}": loss}
