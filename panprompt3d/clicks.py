# Modified for the PanPrompt3D release: packaging/import cleanup.
import numpy as np
import torch


def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []
    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
    to_point_mask = torch.logical_or(fn_masks, fp_masks)
    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False
        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor([int(is_positive)]).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)
    return (batch_points, batch_labels)
