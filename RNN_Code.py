#Quantum Diffusion Model RNN Code
import cupy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Setting up the seed
def seed_setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
seed_setup(0)
#Pauli Basis for 1 Qubit
X = torch.tensor([[0,1],[1,0]], dtype = torch.cfloat, device = device)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype = torch.cfloat, device = device)
Z = torch.tensor([[1,0],[0,1]], dtype = torch.cfloat, device = device)
I2 = torch.eye(2, dtype = torch.cfloat, device=device)
PAULI = [X,Y, Z]
#Initial State Definition, |0>
psi_0 = torch.tensor([1,0], dtype = torch.cfloat, device = device)
#Kraus Operator Definiton
def K_dt(gamma, dt, dot, ot):
    coeff = (2*gamma)/(np.pi * dt)
    exp =-1*(gamma/dt)*(dot-ot*dt)**2
    return torch.tensor((coeff**0.25)*np.exp(exp), dtype = torch.cfloat, device=device)

def stochastic_evolve(K_dt, psi_0):
    num = K_dt @ psi_0
    denom = np.linalg.norm(K_dt @ psi_0)
    return num / denom

def Hamiltonian(h):
    return h[0]*X+h[1]*Y+h[2]*Z

#Fidelity based loss function 
def loss(V, psi_t, psi_t_dt,):
    prod = psi_t_dt @ V @ psi_t
    return 1 - np.mean(np.linalg.norm(prod)**2)

def pauli_exp(psi_t):
    #Equivalent to <psi | P | psi>
    return psi_t @ I2 @ X @ Y @ Z @ psi_t
def find_dot(psi, O, gamma, dt):
    prod = (psi @ O @ psi) * dt
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
    return torch.tensor(prod + (dW)/(2*np.sqrt(gamma)), dtype = torch.cfloat, device=device)
input_dim = 2
embedding_dim = 128
hidden_dim = 128
output_size= 2
def simulate_kraus_traj(psi0, steps = 20, dt = 0.01, gamma=1):
    psi = psi0.clone()
    traj = [psi.clone()]
    zs = [pauli_exp(psi)]
    dos = []
    Ks =[]
    for t in range(steps):
        O = PAULI[np.random.randint(0,3).item()]
        dot = find_dot(psi, O, gamma, dt)
        K = K_dt(gamma, dt, dot, O)
        psi = stochastic_evolve(K, psi)
        traj.append(psi.close())
        zs.append(pauli_exp(psi).cpu())
        dos.append(dot)
        Ks.append(K)
    return traj

class Diffusion_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size):
        super(Diffusion_RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first = True)
        self.fc = nn.LSTM(hidden_dim, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)
        out, _= self.rnn(x,h0)
        out = self.fc(out[:, 1, :])
        return out 
model = Diffusion_RNN(input_dim, embedding_dim, hidden_dim, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    initial_conditions = simulate_kraus_traj(psi_0)
    h = [np.random.randint(0,1),np.random.randint(0,1), np.random.randint(0,1)]
    for j, psi in enumerate(initial_conditions):
        output = model(psi)
        loss = loss(Hamiltonian(h), initial_conditions[j-1], psi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(initial_conditions):.4f}')