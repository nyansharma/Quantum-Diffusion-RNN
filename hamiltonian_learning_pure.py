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
#ODE implementation of diffusion 

def drift_vector(psi, O):
    O_psi = np.dot(O, psi)
    delta_O = O - np.vdot(psi, O_psi)
    return -0.5*np.dot(delta_O**2, psi)
def score_vector(noise, prob, dpsi):
    inside= (prob*(dpsi.conj().T @ noise)*(noise.conj().T)).cong().T
    return (1/prob)*inside
def flow_vector(score, drift):
    return drift - 0.5*score
def Hamiltonian(flow, psi):
    h = 1j*(flow @ psi)
    return h + h.conj().T
def noise_vector(O, psi):
    g = O*psi-psi.T*O
    return g
#Score function, which we use to learn hamiltonian
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
def forward_diffusion(dt, T):
    n = int(T / dt)
    psi = [None] * n
    psi[0] = psi_0.clone()
    scores_gt = [None] * n # Ground truth scores to train the RNN
    
    for i in range(n - 1):
        # 1. Calculate physics-based drift (from your original code)
        O = PAULI[np.random.randint(0, 4)]
        drift = drift_vector(psi[i], O)
        
        # 2. Simulate the stochastic step (Euler-Maruyama)
        # For training, we treat the noise injection as the "target score"
        noise = noise_vector(O, psi[i])
        dW = np.random.normal(0, np.sqrt(dt)) 
        
        # Actual change in state
        dpsi = drift * dt + noise * dW
        psi[i+1] = (psi[i] + dpsi) / np.linalg.norm(psi[i] + dpsi)
        
        # 3. Ground truth score for this step
        # Based on: flow = (dpsi/dt) = drift - 0.5 * score
        # => score = 2 * (drift - (dpsi/dt))
        scores_gt[i] = 2 * (drift - (dpsi / dt))
        
    return psi, scores_gt

class QuantumScoreRNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Encoder to map psi_0 to initial hidden state
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # RNN input: [psi_t_real, psi_t_imag, time]
        self.rnn = nn.LSTM(input_size=5, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4) # Real/Imag for 2-element complex vector

    def forward(self, psi_t, psi_0, t):
        # Encode psi_0 as context
        p0_vec = torch.cat([psi_0.real, psi_0.imag], dim=-1)
        h_0 = self.encoder(p0_vec).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)

        # Prepare psi_t and time
        pt_vec = torch.cat([psi_t.real, psi_t.imag], dim=-1).unsqueeze(1)
        t_tensor = torch.full((pt_vec.size(0), 1, 1), t, device=device)
        rnn_input = torch.cat([pt_vec, t_tensor], dim=-1)

        out, _ = self.rnn(rnn_input, (h_0, c_0))
        score_raw = self.fc(out[:, -1, :])
        
        # Return predicted score as complex64
        return torch.complex(score_raw[:, 0:2], score_raw[:, 2:4])
def fidelity_loss(psi_current, H_pred, psi_prev, dt):
    """
    Loss based on how well the predicted H evolves psi_t back to psi_{t-dt}
    """
    # Reverse evolution: exp(+i * H * dt) 
    # (Note: Forward is usually -i, so reverse is +i)
    V = (1j * H_pred * dt).matrix_exp()
    psi_next_pred = V @ psi_current
    
    # Fidelity |<psi_prev | psi_pred>|^2
    inner_prod = torch.vdot(psi_prev, psi_next_pred)
    return 1 - torch.abs(inner_prod)**2
# --- Training Loop ---

model = QuantumScoreRNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs=100
for epoch in range(num_epochs):
    # Generate trajectory and scores
    sim_psi, sim_scores = forward_diffusion(dt=0.05, T=1.0)
    psi_0 = sim_psi[0].detach()
    
    epoch_loss = 0
    # Training reverse-step by reverse-step
    for i in range(len(sim_psi) - 1, 0, -1):
        psi_t = sim_psi[i].detach()
        target_score = sim_scores[i]
        t_val = i * 0.05
        
        pred_score = model(psi_t, psi_0, t_val)
        
        # Loss: ||pred_score - target_score||^2
        loss = torch.mean(torch.abs(pred_score - target_score)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()