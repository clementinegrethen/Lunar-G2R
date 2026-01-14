import numpy as np
import torch
import torch.nn as nn


class SurrenderMSELoss(nn.Module):
    def __init__(self):
        super(SurrenderMSELoss, self).__init__()
        import brdfgen.config as config
        config.batch_loss = 0

    def forward(self, outputs, gt):
        # doing nothing, actually computed in Surrender custom layer
        import brdfgen.config as config
        return config.batch_loss + outputs[0,0,0,0] - outputs[0,0,0,0]


class RegularisationLoss(nn.Module):
    def __init__(self):
        super(RegularisationLoss, self).__init__()

    def forward(self, outputs):
        return torch.abs(outputs[...,:-1,:] - outputs[...,1:,:]).sum() + torch.abs(outputs[...,:-1] - outputs[...,1:]).sum()


class RotationLoss(nn.Module):
    def __init__(self, channels):
        super(RotationLoss, self).__init__()
        self.channels = channels

    def forward(self, inputs, outputs, model: nn.Module):
        loss = torch.scalar_tensor(0).to(inputs.device)
        for i in range(1, 4):
            N_R_I = model.forward(torch.rot90(inputs, k=i, dims=[2, 3]))
            R_N_I = torch.rot90(outputs, k=i, dims=[2, 3])[:,:self.channels,...]
            loss += torch.abs(R_N_I - N_R_I).sum()
        return loss


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
class DEMStats():

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.min = 1e5
        self.max = -1e5

    def update(self, new_value: np.ndarray):
        minv, maxv = np.min(new_value), np.max(new_value)
        self.min = minv if minv < self.min else self.min
        self.max = maxv if maxv > self.max else self.max
        self.count += new_value.size
        delta = new_value - self.mean
        self.mean += np.sum(delta) / self.count
        delta2 = new_value - self.mean
        self.M2 += np.sum(np.multiply(delta, delta2))

    def get_min(self):
        if self.count < 1:
            return float("nan")
        else:
            return self.min

    def get_max(self):
        if self.count < 1:
            return float("nan")
        else:
            return self.max

    def get_mean(self):
        if self.count < 2:
            return float("nan")
        else:
            return self.mean

    def get_variance(self):
        if self.count < 2:
            return float("nan")
        else:
            return self.M2/self.count

    def get_std(self):
        if self.count < 2:
            return float("nan")
        else:
            return np.sqrt(self.M2 / self.count)
