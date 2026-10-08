# Modified for the PanPrompt3D release: packaging/import cleanup.
import torch.nn as nn
import torch
import torch.nn.functional as F


class CF_Loss_3D(nn.Module):

    def __init__(self, img_depth, beta, gamma):
        super(CF_Loss_3D, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.img_depth = img_depth
        self.CE = nn.CrossEntropyLoss()
        self.p = torch.tensor([img_depth], dtype=torch.float, device="cuda")
        self.n = torch.log(self.p) / torch.log(torch.tensor([2.0], device="cuda"))
        self.n = torch.floor(self.n)
        self.sizes = 2 ** torch.arange(self.n.item(), 1, -1, device="cuda").to(
            dtype=torch.int
        )

    def get_count_3d_binary(self, sizes, masks_pred_softmax):
        counts = torch.zeros((masks_pred_softmax.shape[0], len(sizes)), device="cuda")
        index = 0
        for size in sizes:
            stride = (1, size, size)
            pool = nn.AvgPool3d(kernel_size=(1, size, size), stride=stride)
            if (
                masks_pred_softmax.size(2) >= size
                and masks_pred_softmax.size(3) >= size
            ):
                S = pool(masks_pred_softmax)
                S = S[:, 1, ...]
                count_positive = (S > 0).float().mean()
                counts[..., index] = count_positive
            index += 1
        return counts

    def forward(self, prediction, ground_truth):
        prediction_softmax = F.softmax(prediction, dim=1)
        Loss_vd = torch.abs(
            prediction_softmax[:, 0, ...].sum() - ground_truth[:, 0, ...].sum()
        ) / (
            prediction_softmax.shape[0]
            * prediction_softmax.shape[2]
            * prediction_softmax.shape[3]
            * prediction_softmax.shape[4]
        )
        combined = torch.cat((prediction_softmax, ground_truth), 1)
        counts = self.get_count_3d_binary(self.sizes, combined)
        counts_first_channel = counts[..., 0].unsqueeze(1)
        loss_FD_term = self.sizes.unsqueeze(0) * counts_first_channel**2
        artery_ = torch.sqrt(torch.sum(loss_FD_term, dim=1))
        size_t = torch.sqrt(torch.sum(self.sizes**2))
        loss_FD = artery_ / size_t / prediction_softmax.shape[0]
        loss_value = self.beta * loss_FD + self.gamma * Loss_vd
        return loss_value
