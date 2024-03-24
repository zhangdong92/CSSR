# Operation about deformation field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeformationFieldApplyModule(nn.Module):
    """apply deformation field """
    
    def __init__(self, width, height, device):
        super(DeformationFieldApplyModule, self).__init__()
        self.width = width
        self.height = height
        self.xArray, self.yArray = np.meshgrid(np.arange(width), np.arange(height))
        self.xArray = torch.tensor(self.xArray, requires_grad=False, device=device)
        self.yArray = torch.tensor(self.yArray, requires_grad=False, device=device)
        
    def forward(self, slice0, deformationField):
        """get slice_i, by slice0 and phi_{0i}"""

        # deformationField.shape: [n, sliceChannel，deformationFIeldChannel(u/v), h, w]
        u = deformationField[:, 0, :, :]
        v = deformationField[:, 1, :, :]
        x = self.xArray.unsqueeze(0).expand_as(u).float() + u
        y = self.yArray.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x2 = 2*(x/self.width - 0.5)
        y2 = 2*(y/self.height - 0.5)
        grid = torch.stack((x2,y2), dim=3)
        # apply deformation field on slice0
        slice_i = torch.nn.functional.grid_sample(slice0, grid)
        return slice_i
