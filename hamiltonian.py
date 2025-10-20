import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
    import cupy as np
else:
    device = torch.device("cpu")
    import numpy as np
X = torch.tensor([[0,1],[1,0]], dtype=torch.cfloat, device=device)
Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.cfloat, device=device)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)  
I2 = torch.eye(2, dtype=torch.cfloat, device=device)
PAULI = [I2, X, Y, Z]
