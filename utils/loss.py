"""Custom losses."""

import torch
import torch.nn as nn

__all__ = [
    'MixSoftmaxCrossEntropyLoss',
    'MixSoftmaxCrossEntropyOHEMLoss'
]


class MixSoftmaxCrossEntropyLoss(nn.Module):
    """
    Standard CrossEntropy Loss for semantic segmentation.
    Supports auxiliary outputs.
    """

    def __init__(self, aux=True, aux_weight=0.4, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__()

        self.aux = aux
        self.aux_weight = aux_weight

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_label
        )

    def _aux_forward(self, *inputs):
        *preds, target = inputs

        loss = self.criterion(preds[0], target)

        for pred in preds[1:]:
            loss += self.aux_weight * self.criterion(pred, target)

        return loss

    def forward(self, *inputs):

        preds, target = inputs

        if not isinstance(preds, (tuple, list)):
            preds = (preds,)

        if self.aux:
            return self._aux_forward(*preds, target)

        return self.criterion(preds[0], target)


class MixSoftmaxCrossEntropyOHEMLoss(MixSoftmaxCrossEntropyLoss):
    """
    For this project we disable OHEM and use normal CrossEntropy.
    """

    def __init__(self,
                 aux=False,
                 aux_weight=0.4,
                 ignore_index=-1,
                 **kwargs):

        super().__init__(
            aux=aux,
            aux_weight=aux_weight,
            ignore_label=ignore_index
        )