import torch
import hamiltonian_learning_pure as hamiltonian_learning
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
#nothing strenuous so cpu is fine
device = torch.device("cpu")

# Pauli basis
X = torch.tensor([[0,1],[1,0]], dtype=torch.cfloat, device=device)
Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.cfloat, device=device)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)  
I2 = torch.eye(2, dtype=torch.cfloat, device=device)
PAULI = [I2, X, Y, Z]
T = 1
# Initial state 
def Hamiltonian(h):
    return h[0]*X+h[1]*Y+h[2]*Z
H = Hamiltonian(hamiltonian_learning.H_history[-1])
psi_f = hamiltonian_learning.psi_history[-1]
num_trajs=50
trajectories = []
for _ in range(num_trajs):
    traj = hamiltonian_learning.forward_diffusion(0.5, 0.01, T/2)
    traj = torch.stack(traj)  
    trajectories.append(traj)

forward_traj = torch.stack(trajectories)  
dataset = TensorDataset(forward_traj)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
ham_model = hamiltonian_learning.Diffusion_RNN(2,64, 2).to(device)
ham_model.load_state_dict(torch.load("trained_diffusion_pure_rnn.pth"))
ham_model.eval()
def complex_to_real(x):
    real = torch.view_as_real(x)                  
    real = real.reshape(real.shape[0], real.shape[1], real.shape[2]*2)
    return real

def real_to_complex(x):
    D = x.shape[0] // 2
    x = x.reshape(D, 2)                   
    return torch.view_as_complex(x)      
#Score function
def unitary_score(H, dt):
    return (-1j * H * dt).matrix_exp()
#Loss function (particularly for reverse)
def bridge_loss(psi_mid_fwd, psi_mid_rev):
    return 1 - torch.abs(torch.vdot(psi_mid_fwd, psi_mid_rev))**2
def reverse_step(model, psi, H, dt):
    U = unitary_score(H, dt)
    psi_unitary = U @ psi
    psi_input = psi.unsqueeze(0).unsqueeze(0)               
    psi_input_real = complex_to_real(psi_input)          

    psi_nn = model(psi_input_real).squeeze(0).squeeze(0)  

    psi_prev = psi_unitary + real_to_complex(psi_nn)     
    psi_prev = psi_prev / torch.linalg.norm(psi_prev)
    return psi_prev

def reverse_diffusion(model, H, psi_f, dt, T, t_target):
    n_steps = int((T - t_target)/dt)
    traj = [psi_f]
    psi = psi_f
    for _ in range(n_steps):
        psi = reverse_step(model, psi, H, dt)
        traj.append(psi)
    traj.reverse()
    return traj
class ReverseRNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, psi_seq):
        h, _ = self.rnn(psi_seq)
        pred = self.fc(h)
        return pred
model = ReverseRNN(hidden_size=128, input_size=2*2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
num_epochs = 100
for epoch in range(num_epochs):
    for psi_forward_batch in data_loader:
        psi_forward = psi_forward_batch[0]   
        psi_target = psi_forward[:, :-1, :]
        psi_input  = psi_forward[:, 1:, :]

        # Convert complex → real for RNN
        psi_input_real  = complex_to_real(psi_input)
        psi_target_real = complex_to_real(psi_target)

        pred_real = model(psi_input_real)
        loss = criterion(pred_real, psi_target_real)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
if __name__ == "__main__":
    gamma, dt, T = 0.05, 0.01, 1.0

    psi_fwd = hamiltonian_learning.forward_diffusion(gamma, dt, T/2)
    psi_mid_fwd = psi_fwd[-1]

    psi_T = hamiltonian_learning.forward_diffusion(gamma, dt, T)[-1]

    # reverse half
    psi_rev = reverse_diffusion(model, H, psi_f, dt, T, T/2)
    psi_mid_rev = psi_rev[-1]

    # bridge consistency
    L_bridge = bridge_loss(psi_mid_fwd, psi_mid_rev)
    print(f"Bridge loss (fidelity distance): {L_bridge.item():.6f}")


    T = min(len(psi_fwd), len(psi_rev))  # match lengths
    time = torch.arange(T)

    psi_fwd = torch.stack(psi_fwd)[:T]
    psi_rev = torch.stack(psi_rev)[:T]
    plt.figure(figsize=(8,5))

    for i in range(psi_fwd.shape[1]):  # loop over qubit amplitudes
        plt.plot(time, (torch.abs(psi_fwd[:, i].detach())**2).cpu(), 'o-', label=f'Forward |ψ_{i}|^2')
        plt.plot(time, (torch.abs(psi_rev[:, i].detach())**2).cpu(), 'x--', label=f'Reverse |ψ_{i}|^2')

    plt.xlabel("Time step")
    plt.ylabel("Probability |ψ_i|^2")
    plt.title("Forward vs Reverse Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig("pure_rnn.png")
    plt.show()