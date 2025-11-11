import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
device = torch.device("cpu")

# Pauli basis
X = torch.tensor([[0,1],[1,0]], dtype=torch.cfloat, device=device)
Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.cfloat, device=device)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)  
I2 = torch.eye(2, dtype=torch.cfloat, device=device)
PAULI = [I2, X, Y, Z]

# Initial state 
np.random.seed(67)
x = np.random.randn(2) + 1j * np.random.randn(2)  # random complex vector
psi_0 = torch.tensor(
    x / np.linalg.norm(x), 
    dtype=torch.complex64,  # use .complex64 for CUDA compatibility
    device=device
)
def Hamiltonian(h):
    return h[0]*X+h[1]*Y+h[2]*Z
#Score function
def unitary_score(H, dt):
    return (1j * H * dt).matrix_exp()
#Loss function (particularly for reverse)
def mean_fidelity_loss(psi, psi_dt, model, dt):
    H = Hamiltonian(model.h)
    V = unitary_score(H, dt)
    return 1 - torch.abs(torch.vdot(psi, V @ psi_dt))**2

def pauli_exp(psi, O):
    return torch.conj(psi) @ (O @ psi)

# Simulate forward diffusion via euler maruyama
def forward_diffusion(gamma, dt, T):
    n = int(T / dt)
    psi = [None] * n
    psi[0] = psi_0.clone()
    O = [PAULI[np.random.randint(0, 4)] for _ in range(n)]
    gamma = float(gamma)
    for i in range(n - 1):
        psi_i = psi[i]
        Oi = O[i].to(torch.complex64).to(device)
        exp_O = (psi_i.conj() @ (Oi @ psi_i)).item()
        delta_O = Oi - exp_O * torch.eye(2, dtype=torch.complex64, device=device)
        dW = torch.sqrt(torch.tensor(dt, device=device)) * torch.randn((), device=device)
        dW = dW.to(torch.complex64)
        update = (
            torch.eye(2, dtype=torch.complex64, device=device)
            - (gamma / 2) * delta_O @ delta_O * dt
            + torch.sqrt(torch.tensor(gamma, device=device)) * delta_O * dW
        )
        psi[i+1] = update @ psi_i
        psi[i+1] = psi[i+1] / torch.linalg.norm(psi[i+1])
        psi[i+1] = psi[i+1].clone().detach().requires_grad_(True)
    return psi

input_dim = 2
hidden_dim = 64
output_size= 2
class Diffusion_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(Diffusion_RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim,hidden_size= hidden_dim, batch_first = True, device = device)
        self.fc = nn.Linear(hidden_dim, output_size, device=device)
        self.h = nn.Parameter(torch.randn(3))
    def forward(self, psi_t, dt):
        out, _ = self.rnn(psi_t)
        out = self.fc(out[:, -1, :])
        H = Hamiltonian(self.h)
        V = (1j*H*dt).matrix_exp()
        psi_t_complex = psi_t[:, -1, :].to(torch.complex64)
        correction = out.to(torch.complex64)
        pred_next = (V @ psi_t_complex.T).T + correction 
        pred_next = pred_next / torch.linalg.norm(pred_next, dim=1, keepdim=True)
        return pred_next, V, H
model = Diffusion_RNN(input_dim, hidden_dim, output_size).to(device)
torch.save(model.state_dict(), "trained_diffusion_pure_rnn.pth")
num_epochs = 1000
criterion = nn.CrossEntropyLoss()
H_history = []
psi_history = []
optimizer = optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # simulate forward diffusion
    sim = forward_diffusion(gamma = 0.05, dt = 0.05, T =1)
    for j in range(len(sim)-1):
        # convert psi to tensor with batch and seq dimensions
        psi_t = sim[j]
        psi_dt = sim[j+1]
        psi_tensor = torch.tensor(psi_t, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # forward pass
        output, V, H = model(psi_tensor, 0.05)
        loss = mean_fidelity_loss(psi_t, psi_dt, model, dt=0.05)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        H_history.append(model.h.detach().clone())
        psi_history.append(psi_t.detach().clone())
        total_loss += loss.item()
    psi_history.append(sim[-1])
H_history = torch.stack(H_history)
H_np = H_history.detach().cpu().numpy()
