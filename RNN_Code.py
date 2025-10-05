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
PAULI = [I2, X, Y, Z]

# Initial state 
x = np.random.randn(2).astype(np.complex128)
psi_0 = torch.tensor(x/np.linalg.norm(x), dtype=torch.cfloat, device=device)


# Hamiltonian
def Hamiltonian(h):
    return h[0]*X + h[1]*Y + h[2]*Z
#Score function
def unitary_score(H, dt):
    return (1j * H * dt).matrix_exp()
#Loss function (particularly for reverse)
def mean_fidelity_loss(psi, V, psi_dt):
    return torch.conj(psi) @ (V@psi_dt)

def pauli_exp(psi, O):
    return torch.conj(psi) @ (O @ psi)

# Simulate forward diffusion via euler maruyama
def forward_diffusion(gamma, dt, T):
    n = int(T/dt)
    psi = [None]*n
    psi[0]=psi_0
    O=[]
    gamma=float(gamma)
    for i in range(n):
        O.append(PAULI[np.random.randint(0,4).item()]) #creates an array of observables
    for i in range(n-1):
        delta_ot = torch.tensor(O[i]-pauli_exp(psi[i], O[i]).item(), dtype = torch.complex64, device=device)
        noise = torch.randn(delta_ot.shape, dtype=torch.complex64, device=device)
        psi[i+1] = (-(gamma/2)*delta_ot**2+torch.sqrt(torch.tensor(gamma, device=device))*delta_ot*noise)@psi[i]
    return psi

input_dim = 2
hidden_dim = 128
output_size= 2
class Diffusion_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(Diffusion_RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim,hidden_size= hidden_dim, batch_first = True, device = device)
        self.fc = nn.LSTM(hidden_dim, output_size, device=device)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)
        out, _= self.rnn(x,h0)
        out = self.fc(out[:, 1, :])
        return out 
model = Diffusion_RNN(input_dim, hidden_dim, output_size)
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # simulate forward diffusion
    sim = forward_diffusion(gamma = 0.5, dt = 0.01, T =1)
    h = np.random.randint(0, 2, size=3)  

    for j, psi in enumerate(sim[:-1]):
        # convert psi to tensor with batch and seq dimensions
        psi_tensor = torch.tensor(psi, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # forward pass
        output = model(psi_tensor)

        # compute Hamiltonian and loss
        H = Hamiltonian(h).to(device)
        V=unitary_score(H, dt=0.01)
        L = mean_fidelity_loss(psi[j], V, psi[j+1])

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        epoch_loss += L.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(sim):.4f}")