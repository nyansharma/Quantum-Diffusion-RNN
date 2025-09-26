import torch
import torch.nn as nn
import torch.optim as optim
import random
import cupy as np 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pauli basis
X = torch.tensor([[0,1],[1,0]], dtype=torch.cfloat, device=device)
Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.cfloat, device=device)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)  
I2 = torch.eye(2, dtype=torch.cfloat, device=device)
PAULI = [X, Y, Z]

# Initial state |0>
psi_0 = torch.tensor([1,0], dtype=torch.cfloat, device=device)

# Kraus operator
def K_dt(gamma, dt, dot, O_t):
    prefactor = (2 * gamma / (torch.pi * dt))**0.25
    I = torch.eye(O_t.shape[0], device=device, dtype=O_t.dtype)
    diff = dot * I - O_t * dt
    A = -(gamma / dt) * (diff @ diff)
    return prefactor * torch.linalg.matrix_exp(A)

def stochastic_evolve(K, psi):
    num = K @ psi
    denom = torch.linalg.norm(num)
    return num / denom

# Hamiltonian
def Hamiltonian(h):
    return h[0]*X + h[1]*Y + h[2]*Z

def finding_ground_state(H):
    eigvals = torch.linalg.eigvals(H).real
    return torch.min(eigvals)

def loss(psi, H):
    energy = torch.conj(psi) @ (H @ psi)
    energy = energy.real
    return energy - finding_ground_state(H)

def pauli_exp(psi, O):
    return torch.conj(psi) @ (O @ psi)
def find_dot(psi, O, gamma, dt):
    exp_val = pauli_exp(psi, O).real
    dW = torch.sqrt(torch.tensor(dt, device=device)) * torch.randn(1, device=device)
    return exp_val * dt + dW / (2*torch.sqrt(torch.tensor(gamma, device=device)))

# Simulate trajectory
def simulate_kraus_traj(psi0, steps=20, dt=0.01, gamma=1.0):
    psi = psi0.clone()
    traj = [psi.clone()]
    for t in range(steps):
        O = PAULI[random.randint(0,2)]
        dot = find_dot(psi, O, gamma, dt)
        K = K_dt(gamma, dt, dot, O)
        psi = stochastic_evolve(K, psi)
        traj.append(psi.clone())
    return traj
input_dim = 2
embedding_dim = 128
hidden_dim = 128
output_size= 2
def simulate_kraus_traj(psi0, steps = 20, dt = 0.01, gamma=1):
    psi = psi0.clone()
    traj = [psi.clone()]
    O = PAULI[np.random.randint(0,3).item()]
    zs = [pauli_exp(psi,O)]
    dos = []
    Ks =[]
    for t in range(steps):
        O = PAULI[np.random.randint(0,3).item()]
        dot = find_dot(psi, O, gamma, dt)
        K = K_dt(gamma, dt, dot, O)
        psi = stochastic_evolve(K, psi)
        traj.append(psi.clone())
        zs.append(pauli_exp(psi,O))
        dos.append(dot)
        Ks.append(K)
    return traj

class Diffusion_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_size):
        super(Diffusion_RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, device=device)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first = True, device = device)
        self.fc = nn.LSTM(hidden_dim, output_size, device=device)
    
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
        output = model(torch.tensor(j, dtype=torch.long, device = device))
        loss = loss(Hamiltonian(h), psi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(initial_conditions):.4f}')