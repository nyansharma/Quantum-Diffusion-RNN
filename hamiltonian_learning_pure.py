import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
device = torch.device("cpu")

# Pauli basis
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)  
I2 = np.eye(2, dtype=complex)
PAULI = [I2, X, Y, Z]

# Initial state 
np.random.seed(67)
x = np.random.randn(2) + 1j * np.random.randn(2)
psi_0 = np.array(x / np.linalg.norm(x), dtype=complex)

def complex_to_real(psi):
    return torch.cat([psi.real, psi.imag], dim=-1)

def real_to_complex(x):
    re = x[..., :2]
    im = x[..., 2:]
    c = torch.complex(re, im)
    # Normalize to keep on unit sphere
    norm = torch.sqrt(torch.sum(c.real**2 + c.imag**2, dim=-1, keepdim=True))
    return c / (norm + 1e-8)

# Fokker-Planck equation for probability evolution
def fokker_planck_update(rho, psi, O, dt):
    """
    Implements: ∂_t ρ(ψ) = -⟨∂_ψ | (f(ψ) ρ(ψ))⟩ + (1/2)⟨∂_ψ | (g(ψ) ρ(ψ) ⟨g(ψ)|)| ∂_ψ⟩
    
    where:
    - f(ψ) is the drift vector (deterministic part)
    - g(ψ) is the diffusion matrix (stochastic part)
    - ρ(ψ) is the probability density
    """
    # Drift term: f(ψ)
    f_psi = drift_vector(psi, O)
    
    # Diffusion term: g(ψ)
    g_psi = noise_vector(O, psi)
    
    # Drift contribution: -∇·(f ρ)
    # Approximate divergence numerically
    eps = 1e-6
    drift_contrib = 0.0
    
    for i in range(2):
        psi_plus = psi.copy()
        psi_plus[i] += eps
        psi_plus = psi_plus / np.linalg.norm(psi_plus)
        
        f_plus = drift_vector(psi_plus, O)
        drift_contrib -= np.real((f_plus[i] - f_psi[i]) / eps) * rho
    
    # Diffusion contribution: (1/2) ∇·(g g† ρ ∇)
    # This is the "spreading" term - proportional to second derivative
    diffusion_contrib = 0.0
    
    for i in range(2):
        for j in range(2):
            # Second derivative approximation
            psi_plus = psi.copy()
            psi_minus = psi.copy()
            psi_plus[i] += eps
            psi_minus[i] -= eps
            
            psi_plus = psi_plus / np.linalg.norm(psi_plus)
            psi_minus = psi_minus / np.linalg.norm(psi_minus)
            
            # g†g element
            D_ij = np.real(np.conj(g_psi[i]) * g_psi[j])
            
            # Second derivative of ρ
            if i == j:
                # Simple finite difference for second derivative
                d2_rho = (rho - rho) / (eps * eps)  # Will be updated with neighboring values
            
            diffusion_contrib += 0.5 * D_ij * d2_rho
    
    # Update probability
    drho_dt = drift_contrib + diffusion_contrib
    
    return rho + drho_dt * dt

# Enhanced forward diffusion with probability tracking
def forward_diffusion_with_probability(dt, T, noise_strength=0.5):
    """
    Simulate quantum state diffusion with explicit probability tracking
    using both the SDE for states and Fokker-Planck for probability
    """
    n = int(T / dt)
    psi = np.zeros((n, 2), dtype=np.complex128)
    psi[0] = psi_0.astype(np.complex128)
    rho = np.ones(n, dtype=np.float64)  # Probability at each time step
    rho[0] = 1.0  # Start with probability 1
    H = np.zeros((n, 2, 2), dtype=np.complex128)
    
    for i in range(n - 1):
        # Select random measurement operator
        O = PAULI[np.random.randint(0, 4)]
        
        # Deterministic drift
        drift = drift_vector(psi[i], O)
        
        # Stochastic noise
        dW_real = np.random.randn(2) * np.sqrt(dt)
        dW_imag = np.random.randn(2) * np.sqrt(dt)
        dW = dW_real + 1j * dW_imag
        
        # Noise vector
        noise = noise_vector(O, psi[i])
        
        # Time-dependent noise strength
        current_noise_strength = noise_strength * (i / n)
        
        # SDE update for state
        delta_psi = drift * dt + current_noise_strength * noise * np.linalg.norm(dW)
        psi[i+1] = psi[i] + delta_psi
        psi[i+1] = psi[i+1] / np.linalg.norm(psi[i+1])
        
        # Fokker-Planck update for probability
        rho[i+1] = fokker_planck_update(rho[i], psi[i], O, dt)
        
        # Ensure probability stays positive and normalized
        rho[i+1] = max(rho[i+1], 1e-10)
        
        # Compute effective Hamiltonian
        H[i] = Hamiltonian(drift, psi[i])
    
    # Normalize probability distribution
    rho = rho / np.sum(rho)
    
    return psi, H, rho

# Physics definitions of the ODE dynamics
def drift_vector(psi, O):
    psi = psi.flatten()
    O_psi = O@psi
    exp_O = np.dot(psi.conj(), O_psi)
    delta_O = O - exp_O*np.eye(2, dtype=complex)
    return -0.5*((delta_O@delta_O)@psi)

def noise_vector(O, psi):
    avg_O = np.vdot(psi, O @ psi)
    delta_O = O - avg_O * np.eye(2)
    return delta_O @ psi

def score_vector(noise, prob, dpsi):
    inside = (prob*(dpsi.conj().T @ noise)*(noise.conj().T)).conj().T
    return (1/prob)*inside

def flow_vector(score, drift):
    return drift - 0.5*score

def Hamiltonian(flow, psi):
    h = 1j*(np.outer(flow, psi))
    return h + h.conj().T

# Simulate forward diffusion via SDE with proper stochastic noise
def forward_diffusion(dt, T, noise_strength=0.5):
    """
    Simulate quantum state diffusion using stochastic differential equation:
    d|ψ⟩ = drift * dt + noise * dW
    where dW is a Wiener process (Brownian motion)
    """
    n = int(T / dt)
    psi = np.zeros((n, 2), dtype=np.complex128)
    psi[0] = psi_0.astype(np.complex128)
    H = np.zeros((n, 2, 2), dtype=np.complex128)
    
    for i in range(n - 1):
        # Select random measurement operator
        O = PAULI[np.random.randint(0, 4)]
        
        # Deterministic drift (Lindblad-like evolution)
        drift = drift_vector(psi[i], O)
        
        # Stochastic noise term (measurement backaction)
        # Sample from Wiener process: dW ~ N(0, dt)
        dW_real = np.random.randn(2) * np.sqrt(dt)
        dW_imag = np.random.randn(2) * np.sqrt(dt)
        dW = dW_real + 1j * dW_imag
        
        # Noise vector (simplified measurement backaction)
        noise = noise_vector(O, psi[i])
        
        # Time-dependent noise strength (increases over time)
        current_noise_strength = noise_strength * (i / n)
        
        # SDE update: d|ψ⟩ = drift*dt + noise*dW
        delta_psi = drift * dt + current_noise_strength * noise * np.linalg.norm(dW)
        
        psi[i+1] = psi[i] + delta_psi
        
        # Renormalize to stay on unit sphere (quantum states must have norm 1)
        psi[i+1] = psi[i+1] / np.linalg.norm(psi[i+1])
        
        # Compute effective Hamiltonian
        flow = drift  # Simplified for numerical stability
        H[i] = Hamiltonian(flow, psi[i])
    
    return psi, H

class ScoreBasedReverseRNN(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encode current state and time to half hidden_dim each
        self.state_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # LSTM for temporal dynamics - input is hidden_dim (state + time concatenated)
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Initialize hidden state projection
        self.h_init = nn.Linear(hidden_dim, hidden_dim)
        
        # Predict the score (gradient of log probability)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )
        
        # Predict the denoising direction
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )
    
    def forward(self, psi_T, t_start, steps, return_trajectory=True):
        if isinstance(psi_T, np.ndarray):
            psi_T = torch.from_numpy(psi_T)
        
        psi_T = psi_T.to(torch.complex64)
        
        # Start from the input state
        curr = complex_to_real(psi_T)
        
        # Initialize LSTM hidden state
        state_enc = self.state_encoder(curr)
        time_enc = self.time_encoder(torch.tensor([[t_start]], dtype=torch.float32)).squeeze(0)
        
        # Concatenate state and time encodings
        combined = torch.cat([state_enc, time_enc], dim=-1)  # Now this is hidden_dim
        h = self.h_init(combined)
        c = torch.zeros(self.hidden_dim)
        
        traj = [curr]
        
        for i in range(steps):
            t = torch.tensor([[t_start - (i+1)/steps]], dtype=torch.float32)
            
            # Encode current state and time
            state_enc = self.state_encoder(curr)
            time_enc = self.time_encoder(t).squeeze(0)
            
            # Combine for RNN input (hidden_dim total)
            rnn_input = torch.cat([state_enc, time_enc], dim=-1).unsqueeze(0)
            
            # Update LSTM - now dimensions match
            h, c = self.rnn(rnn_input, (h.unsqueeze(0), c.unsqueeze(0)))
            h, c = h.squeeze(0), c.squeeze(0)
            
            # Predict score (gradient direction)
            score = self.score_head(h)
            
            # Predict denoising direction
            denoise_direction = self.denoise_head(h)
            
            # Reverse SDE step: moves against noise
            # The update includes both drift reversal and score term
            step_size = 0.02 * (1 - i/steps)  # Adaptive step size
            curr = curr + step_size * (denoise_direction + 0.5 * score)
            
            # Renormalize
            curr_complex = real_to_complex(curr)
            curr = complex_to_real(curr_complex)
            
            if return_trajectory:
                traj.append(curr)
        
        if return_trajectory:
            return torch.stack(traj)
        else:
            return curr

# Enhanced loss that looks at trajectory
class TrajectoryFidelityLoss(nn.Module):
    def __init__(self, trajectory_weight=0.3):
        super().__init__()
        self.trajectory_weight = trajectory_weight
    
    def forward(self, pred_traj, target, intermediate_states=None):
        target = target.to(torch.complex64)
        
        # Final state loss (most important)
        final_pred = pred_traj[-1]
        pred_c = real_to_complex(final_pred)
        overlap = torch.sum(pred_c.conj() * target)
        final_fidelity = torch.abs(overlap) ** 2
        final_loss = 1 - final_fidelity
        
        # Trajectory smoothness loss
        trajectory_loss = 0.0
        if len(pred_traj) > 1:
            for i in range(len(pred_traj) - 1):
                diff = pred_traj[i+1] - pred_traj[i]
                trajectory_loss += torch.sum(diff ** 2)
            trajectory_loss = trajectory_loss / (len(pred_traj) - 1)
        
        # Optional: intermediate state loss if provided
        intermediate_loss = 0.0
        if intermediate_states is not None:
            n_intermediate = min(5, len(pred_traj) // 2)
            indices = torch.linspace(0, len(pred_traj)-1, n_intermediate).long()
            for idx in indices:
                if idx < len(intermediate_states):
                    pred_state = real_to_complex(pred_traj[idx])
                    true_state = torch.from_numpy(intermediate_states[idx]).to(torch.complex64)
                    overlap = torch.sum(pred_state.conj() * true_state)
                    intermediate_loss += 1 - torch.abs(overlap) ** 2
            intermediate_loss = intermediate_loss / n_intermediate
        
        # Combined loss
        total_loss = final_loss + \
                     self.trajectory_weight * trajectory_loss + \
                     0.1 * intermediate_loss
        
        return total_loss
EPOCHS = 100
model = ScoreBasedReverseRNN(hidden_dim=256)
criterion = TrajectoryFidelityLoss(trajectory_weight=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training Step with intermediate states
def train_step(model, optimizer, psi_trajectory, criterion, steps):
    optimizer.zero_grad()
    
    psi_noisy = psi_trajectory[-1]  # End state (noisy)
    psi_clean = psi_trajectory[0]   # Start state (clean)
    
    # Convert to tensors
    psi_noisy = torch.from_numpy(psi_noisy).to(torch.complex64)
    psi_clean = torch.from_numpy(psi_clean).to(torch.complex64)
    
    # Get predicted trajectory
    pred_traj = model(psi_noisy, t_start=1.0, steps=steps)
    
    # Sample some intermediate states for supervision
    n_intermediate = min(5, len(psi_trajectory) // 10)
    indices = np.linspace(0, len(psi_trajectory)-1, n_intermediate, dtype=int)
    intermediate_states = [psi_trajectory[idx] for idx in indices]
    
    # Compute loss
    loss = criterion(pred_traj, psi_clean, intermediate_states)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

print(f"Starting training on {device}...")

EPOCHS = 150
DT = 0.01
T = 1.0
STEPS = int(T / DT)

best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0.0
    for _ in range(20):
        # Generate full trajectory for training
        if np.random.rand() < 0.5:
            psi_fwd, H_fwd, rho_fwd = forward_diffusion_with_probability(dt=DT, T=T, noise_strength=1.0)
        else:
            psi_fwd, H_fwd = forward_diffusion(dt=DT, T=T, noise_strength=1.0)
        
        total_loss += train_step(model, optimizer, psi_fwd, criterion, steps=STEPS)
    
    scheduler.step()
    avg_loss = total_loss / 20
    
    if avg_loss < best_loss:
        best_loss = avg_loss
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {avg_loss:.6f}, Best = {best_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

def reverse_diffusion(model, psi_T, steps):
    model.eval()
    with torch.no_grad():
        traj = model(psi_T, 1.0, steps=steps, return_trajectory=True)
        traj = traj.cpu()
        traj_complex = torch.stack([real_to_complex(t) for t in traj])
    return traj_complex.numpy()

# Test with matching step counts
psi_fwd, H_fwd = forward_diffusion(dt=DT, T=T, noise_strength=1.0)
print(f"\nForward diffusion: {len(psi_fwd)} states")
psi_rev = reverse_diffusion(model, psi_fwd[-1], steps=STEPS)
print(f"Reverse diffusion: {len(psi_rev)} states")

def bloch_coords(psi):
    x = 2 * np.real(np.conj(psi[:,0]) * psi[:,1])
    y = 2 * np.imag(np.conj(psi[:,1]) * psi[:,0])
    z = np.abs(psi[:,0])**2 - np.abs(psi[:,1])**2
    return x, y, z

xf, yf, zf = bloch_coords(psi_fwd)
xr, yr, zr = bloch_coords(psi_rev)

# Compute final fidelity
final_overlap = np.vdot(psi_rev[-1], psi_fwd[0])
final_fidelity = np.abs(final_overlap)**2
print(f"\nFinal fidelity: {final_fidelity:.6f}")

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_sphere_surface(ax, alpha=0.3):
    """Draw a solid Bloch sphere surface"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Smooth shaded sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=alpha, color='lightgray', 
                    linewidth=0, antialiased=True, shade=True, zorder=1)

def add_pole_labels(ax):
    """Add |0⟩ and |1⟩ labels"""
    ax.scatter([0], [0], [1], color='black', s=80, alpha=0.7, zorder=10)
    ax.scatter([0], [0], [-1], color='black', s=80, alpha=0.7, zorder=10)
    ax.text(-0.15, 0, 1.15, '|0⟩', fontsize=14, fontweight='bold', ha='center')
    ax.text(-0.15, 0, -1.15, '|1⟩', fontsize=14, fontweight='bold', ha='center')

# Generate multiple trajectories for visualization with probability weights
np.random.seed(42)
n_trajectories = 500

# Store all forward trajectories with probabilities
all_trajectories = []
all_probabilities = []

for _ in range(n_trajectories):
    result = forward_diffusion_with_probability(dt=DT, T=T, noise_strength=1.5)
    if len(result) == 3:
        psi_traj, _, rho_traj = result
        all_probabilities.append(rho_traj)
    else:
        psi_traj, _ = result
        all_probabilities.append(np.ones(len(psi_traj)))
    all_trajectories.append(psi_traj)

# Select time points to visualize
time_indices = [0, 3, 10, 30, 99]  # Corresponds to t=0, 0.03, 0.1, 0.3, 1.0
time_values = [i * DT for i in time_indices]

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 4))

for idx, t_idx in enumerate(time_indices):
    ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
    
    # Draw sphere
    plot_bloch_sphere_surface(ax, alpha=0.25)
    add_pole_labels(ax)
    
    # Extract states at this time point from all trajectories
    states_at_t = np.array([traj[t_idx] for traj in all_trajectories])
    probs_at_t = np.array([prob[t_idx] for prob in all_probabilities])
    
    x_t, y_t, z_t = bloch_coords(states_at_t)
    
    # Determine coloring and sizing based on probability
    if t_idx == 0:
        # Initial state - single point, make it red
        colors = 'red'
        sizes = 100
        alpha = 1.0
    else:
        # Spread states - color and size by probability
        # Normalize probabilities for visualization
        prob_normalized = probs_at_t / (np.max(probs_at_t) + 1e-10)
        colors = plt.cm.Reds(0.3 + 0.6 * prob_normalized)
        sizes = 10 + 40 * prob_normalized  # Larger dots for higher probability
        alpha = 0.6
    
    # Plot points
    ax.scatter(x_t, y_t, z_t, c=colors, s=sizes, alpha=alpha, 
               edgecolors='darkred', linewidths=0.3, zorder=5)
    
    # Formatting
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_title(f't = {time_values[idx]:.2f}', fontsize=16, fontweight='bold', pad=10)
    
    # Remove axis labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set consistent viewing angle
    ax.view_init(elev=20, azim=30)

plt.tight_layout()
plt.savefig('bloch_sphere_diffusion.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Also create reverse diffusion visualization
print("\nGenerating reverse diffusion visualization...")

# Generate multiple reverse trajectories
all_reverse_trajectories = []
for _ in range(100):
    psi_fwd, _ = forward_diffusion(dt=DT, T=T, noise_strength=1.5)
    psi_rev = reverse_diffusion(model, psi_fwd[-1], steps=STEPS)
    all_reverse_trajectories.append(psi_rev)

# Create reverse diffusion plot
fig2 = plt.figure(figsize=(20, 4))

for idx, t_idx in enumerate([99, 70, 50, 30, 0]):  # Reverse order: t=1.0 to t=0
    ax = fig2.add_subplot(1, 5, idx + 1, projection='3d')
    
    plot_bloch_sphere_surface(ax, alpha=0.25)
    add_pole_labels(ax)
    
    # Extract reverse states
    states_at_t = np.array([traj[t_idx] for traj in all_reverse_trajectories])
    x_t, y_t, z_t = bloch_coords(states_at_t)
    
    if t_idx == 99:
        colors = 'red'
        sizes = 100
        alpha = 1.0
    elif t_idx == 0:
        colors = 'green'
        sizes = 100
        alpha = 1.0
    else:
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(x_t)))
        sizes = 15
        alpha = 0.6
    
    ax.scatter(x_t, y_t, z_t, c=colors, s=sizes, alpha=alpha,
               edgecolors='black', linewidths=0.3, zorder=5)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_title(f't = {t_idx * DT:.2f}', fontsize=16, fontweight='bold', pad=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=20, azim=30)

plt.tight_layout()
plt.savefig('bloch_sphere_reverse_diffusion.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Visualizations saved!")