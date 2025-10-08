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

# Pauli basis
X = torch.tensor([[0,1],[1,0]], dtype=torch.cfloat, device=device)
Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.cfloat, device=device)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)  
I2 = torch.eye(2, dtype=torch.cfloat, device=device)
PAULI = [I2, X, Y, Z]

# Initial state 
x = np.random.randn(2) + 1j * np.random.randn(2)  # random complex vector
psi_0 = torch.tensor(
    x / np.linalg.norm(x), 
    dtype=torch.complex64,  # use .complex64 for CUDA compatibility
    device=device
)

#Score function
def unitary_score(H, dt):
    return (1j * H * dt).matrix_exp()
#Loss function (particularly for reverse)
def mean_fidelity_loss(psi, V, psi_dt):
    return 1 - torch.abs(torch.vdot(psi, V @ psi_dt))**2

def pauli_exp(psi, O):
    return torch.conj(psi) @ (O @ psi)

# Simulate forward diffusion via euler maruyama
def forward_diffusion(gamma, dt, T):
    n = int(T / dt)
    psi = [None] * n
    psi[0] = psi_0.clone()
    O = [PAULI[np.random.randint(0, 4).item()] for _ in range(n)]
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
        H = self.h[0]*X+self.h[1]*Y+self.h[2]*Z
        V = (1j*H*dt).matrix_exp()
        psi_t_complex = psi_t[:, -1, :].to(torch.complex64)
        correction = out.to(torch.complex64)
        pred_next = (V @ psi_t_complex.T).T + correction  # (batch, 2)
        pred_next = pred_next / torch.linalg.norm(pred_next, dim=1, keepdim=True)
        return pred_next, V, H
model = Diffusion_RNN(input_dim, hidden_dim, output_size).to(device)
num_epochs = 1000
criterion = nn.CrossEntropyLoss()
H_history = []
optimizer = optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # simulate forward diffusion
    sim = forward_diffusion(gamma = 0.5, dt = 0.01, T =1)
    for j in range(len(sim)-1):
        # convert psi to tensor with batch and seq dimensions
        psi_t = sim[j]
        psi_dt = sim[j+1]
        psi_tensor = torch.tensor(psi_t, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # forward pass
        output, V, H = model(psi_tensor, 0.01)
        L = mean_fidelity_loss(psi_t, V, psi_dt)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        H_history.append(model.h.detach().clone())
        total_loss += L.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(sim):.4f}, h = {model.h}")
H_history = torch.stack(H_history)
H_np = H_history.detach().cpu().numpy()
print(H_np[len(H_np)-1])
plt.figure(figsize=(8,5))
plt.plot(H_np[:,0], label='h_x')
plt.plot(H_np[:,1], label='h_y')
plt.plot(H_np[:,2], label='h_z')
plt.xlabel('Epoch')
plt.ylabel('Coefficient value')
plt.title('Learned Hamiltonian coefficients (GPU-trained)')
plt.legend()
plt.savefig("Hamiltonian_Chart.png")
plt.show()