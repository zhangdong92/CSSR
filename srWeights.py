# calculate super resolution weights
import torch
import numpy as np


srMulti=8

def calcDfWeight (iTensor, device):
    # example: (when srMulti=8) 0.125,0.25,0.375,0.5,0.625,0.75,0.875
    linspaceArray = np.linspace(1/srMulti, (srMulti-1)/srMulti, srMulti-1)

    i = iTensor.detach().numpy()
    w0 = - (1 - (linspaceArray[i])) * (linspaceArray[i])
    w1 = - (1 - (linspaceArray[i])) * (linspaceArray[i])
    w2 = (linspaceArray[i]) * (linspaceArray[i])
    w3 = (1 - (linspaceArray[i])) * (1 - (linspaceArray[i]))
    return torch.Tensor(w0)[None, None, None, :].permute(3, 0, 1, 2).to(device),\
           torch.Tensor(w2)[None, None, None, :].permute(3, 0, 1, 2).to(device),\
           torch.Tensor(w3)[None, None, None, :].permute(3, 0, 1, 2).to(device),\
           torch.Tensor(w1)[None, None, None, :].permute(3, 0, 1, 2).to(device)

def calcFusionWeight (indices, device):
    linspaceArray = np.linspace(1/srMulti, (srMulti-1)/srMulti, srMulti-1)
    i = indices.detach().numpy()
    w = linspaceArray[i]
    return torch.Tensor(1-w)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(w)[None, None, None, :].permute(3, 0, 1, 2).to(device)

