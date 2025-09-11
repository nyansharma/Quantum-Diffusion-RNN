#Quantum Diffusion Model RNN Code
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
device = torch.device("cuda.0" if torch.cuda.is_available() else "cpu")