import torch
import hamiltonian_learning
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
print(H)
psi_f = hamiltonian_learning.psi_history[-1]
print(psi_f)
forward_traj = hamiltonian_learning.forward_diffusion(0.5, 0.01, T/2)
model = hamiltonian_learning.Diffusion_RNN(2,64, 2).to(device)
model.load_state_dict(torch.load("trained_diffusion_rnn.pth"))
model.eval()
#Score function
def unitary_score(H, dt):
    return (-1j * H * dt).matrix_exp()
#Loss function (particularly for reverse)
def bridge_loss(psi_mid_fwd, psi_mid_rev):
    return 1 - torch.abs(torch.vdot(psi_mid_fwd, psi_mid_rev))**2

def reverse_diffusion(model,H, dt, T):
    n = int(T/dt)
    reverse_traj = [None]*n
    reverse_traj[n-1] = psi_f
    for i in range(1, n-1):
        V =  unitary_score(H, dt)
        reverse_traj[n-1-i] = (V@reverse_traj[n-i])
        reverse_traj[n-1-i]=reverse_traj[n-1-i]/torch.linalg.norm(reverse_traj[n-1-i])
    return reverse_traj
if __name__ == "__main__":
    gamma, dt, T = 0.05, 0.01, 1.0

    psi_fwd = hamiltonian_learning.forward_diffusion(gamma, dt, T/2)
    psi_mid_fwd = psi_fwd[-1]

    psi_T = hamiltonian_learning.forward_diffusion(gamma, dt, T)[-1]

    # reverse half
    psi_rev = reverse_diffusion(model, H, dt, T/2)
    psi_mid_rev = psi_rev[-1]

    # bridge consistency
    L_bridge = bridge_loss(psi_mid_fwd, psi_mid_rev)
    print(f"Bridge loss (fidelity distance): {L_bridge.item():.6f}")