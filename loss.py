import torch 
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B =2, C=20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, bestbox = torch.max(ious, dim = 0)   # bestbox = 0, 1
        exists_box = target[..., 20].unsqueeze(3)   # i.e. if exists an object in the cell

        # for box coords 
        box_predictions = exists_box * ( # i.e. if we have an object
            (
                bestbox * predictions[..., 26:30] 
                + (1 - bestbox) * predictions[..., 21:25]
                # i.e. we select what box is responsible
            )
        )
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
