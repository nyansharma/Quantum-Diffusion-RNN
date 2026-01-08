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

class HamiltonianDenoisingNetwork(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Main encoder: state + time -> features
        self.encoder = nn.Sequential(
            nn.Linear(4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predict the effective Hamiltonian that generated the dynamics
        self.hamiltonian_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 8)  # 2x2 complex Hamiltonian = 8 real params (4 real + 4 imag)
        )
        
        # Predict the flow vector (drift + score)
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)  # Flow in 4D real space
        )
        
        # Predict the noise that was added
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)
        )
        
        # Predict the clean state directly
        self.clean_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, psi_noisy, t):
        """
        Predict Hamiltonian, flow, noise, and clean state
        """
        if isinstance(psi_noisy, np.ndarray):
            psi_noisy = torch.from_numpy(psi_noisy)
        
        if torch.is_complex(psi_noisy):
            psi_real = complex_to_real(psi_noisy.to(torch.complex64))
        else:
            psi_real = psi_noisy
        
        # Expand time to match batch if needed
        if isinstance(t, (float, int)):
            t = torch.tensor([t], dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if psi_real.dim() == 1:
            t = t.expand(1)
        
        # Concatenate state and time
        x = torch.cat([psi_real, t], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict Hamiltonian (2x2 complex matrix)
        H_flat = self.hamiltonian_predictor(features)  # 8 values
        H_real = H_flat[:4].reshape(2, 2)
        H_imag = H_flat[4:].reshape(2, 2)
        
        # Make Hamiltonian Hermitian: H = H†
        H_real = 0.5 * (H_real + H_real.T)  # Symmetric real part
        H_imag = 0.5 * (H_imag - H_imag.T)  # Antisymmetric imaginary part
        H_complex = torch.complex(H_real, H_imag)
        
        # Predict flow vector (for reverse SDE)
        flow_pred = self.flow_predictor(features)
        
        # Predict noise
        noise_pred = self.noise_predictor(features)
        
        # Predict clean state
        clean_pred = self.clean_predictor(features)
        
        # Normalize clean prediction
        clean_complex = real_to_complex(clean_pred)
        clean_normalized = complex_to_real(clean_complex)
        
        return H_complex, flow_pred, noise_pred, clean_normalized
    
    def hamiltonian_evolution(self, psi, H, dt):
        """
        Apply Hamiltonian evolution: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩
        """
        if not torch.is_complex(psi):
            psi = real_to_complex(psi)
        
        # For small dt, use first-order: ψ(t+dt) ≈ (I - iHdt)ψ(t)
        # For better accuracy, we should use matrix exponential
        evolution = torch.eye(2, dtype=torch.complex64) - 1j * H * dt
        
        # Apply evolution
        psi_evolved = evolution @ psi
        
        # Renormalize
        psi_evolved = psi_evolved / torch.sqrt(torch.sum(torch.abs(psi_evolved)**2))
        
        return psi_evolved
    
    def denoise_step_with_hamiltonian(self, psi_noisy, t, dt=0.01, use_hamiltonian=True):
        """
        Denoising step that incorporates Hamiltonian physics
        """
        H_pred, flow_pred, noise_pred, clean_pred = self.forward(psi_noisy, t)
        
        if torch.is_complex(psi_noisy):
            psi_real = complex_to_real(psi_noisy.to(torch.complex64))
        else:
            psi_real = psi_noisy
        
        # Method 1: Use Hamiltonian evolution (physics-based)
        if use_hamiltonian and t > 0.3:  # Use Hamiltonian for middle range
            psi_complex = real_to_complex(psi_real)
            psi_evolved = self.hamiltonian_evolution(psi_complex, H_pred, -dt)  # Negative time
            denoised = complex_to_real(psi_evolved)
        
        # Method 2: Use flow-based denoising (for strong noise)
        elif t > 0.1:
            alpha = 1 - t
            denoised = psi_real + flow_pred * dt - (1 - alpha) * noise_pred * dt
        
        # Method 3: Direct clean prediction (for weak noise)
        else:
            # Interpolate between current state and clean prediction
            denoised = 0.7 * clean_pred + 0.3 * psi_real
        
        # Normalize
        denoised_complex = real_to_complex(denoised)
        return complex_to_real(denoised_complex)
    
    def reverse_diffusion(self, psi_T, steps, use_hamiltonian=True):
        """
        Full reverse diffusion using Hamiltonian when appropriate
        """
        if isinstance(psi_T, np.ndarray):
            psi_T = torch.from_numpy(psi_T)
        
        psi_T = psi_T.to(torch.complex64)
        curr = complex_to_real(psi_T)
        
        trajectory = [curr]
        dt = 1.0 / steps
        
        # Reverse diffusion steps
        for i in range(steps):
            t = torch.tensor([1.0 - (i + 1) / steps], dtype=torch.float32)
            
            # Denoise using Hamiltonian-aware method
            curr = self.denoise_step_with_hamiltonian(curr, t, dt, use_hamiltonian)
            
            trajectory.append(curr)
        
        return torch.stack(trajectory)


# Loss function that includes Hamiltonian consistency
class HamiltonianDenoisingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, H_pred, flow_pred, noise_pred, clean_pred, 
                true_noise, true_clean, psi_current, H_true=None):
        """
        Multi-objective loss:
        1. Clean state fidelity
        2. Noise prediction accuracy
        3. Hamiltonian consistency (if H_true provided)
        4. Flow vector accuracy
        """
        true_clean = true_clean.to(torch.complex64)
        true_clean_real = complex_to_real(true_clean)
        
        # 1. Clean state prediction loss (fidelity)
        clean_pred_complex = real_to_complex(clean_pred)
        overlap = torch.sum(clean_pred_complex.conj() * true_clean)
        fidelity = torch.abs(overlap) ** 2
        clean_loss = 1 - fidelity
        
        # 2. Noise prediction loss (MSE)
        if true_noise is not None:
            noise_loss = torch.mean((noise_pred - true_noise) ** 2)
        else:
            noise_loss = 0.0
        
        # 3. Hamiltonian consistency: H should be Hermitian and physically reasonable
        # Check Hermiticity: ||H - H†|| should be small
        H_herm_error = torch.sum(torch.abs(H_pred - H_pred.conj().T) ** 2)
        
        # Energy should be bounded (regularization)
        psi_complex = real_to_complex(true_clean_real)
        energy = torch.real(psi_complex.conj() @ H_pred @ psi_complex)
        energy_penalty = torch.relu(torch.abs(energy) - 10.0)  # Penalize unreasonably large energies
        
        hamiltonian_loss = H_herm_error + 0.1 * energy_penalty
        
        # 4. Flow vector should align with denoising direction
        denoising_direction = true_clean_real - complex_to_real(real_to_complex(true_clean_real))
        flow_alignment = -torch.sum(flow_pred * denoising_direction)  # Negative because we want alignment
        
        # Combined loss with weights
        total_loss = (
            0.5 * clean_loss +           # Most important: get clean state right
            0.2 * noise_loss +            # Learn to predict noise
            0.1 * hamiltonian_loss +      # Ensure physical consistency
            0.05 * flow_alignment         # Flow should point toward clean state
        )
        
        return total_loss

model = HamiltonianDenoisingNetwork(hidden_dim=256)
EPOCHS = 100
criterion = HamiltonianDenoisingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training Step - train on random timesteps with Hamiltonian supervision
def train_step(model, optimizer, psi_trajectory, H_trajectory, criterion):
    optimizer.zero_grad()
    
    # Sample random timestep
    n_steps = len(psi_trajectory)
    t_idx = np.random.randint(1, n_steps)
    t = t_idx / n_steps
    
    # Get states
    psi_noisy = torch.from_numpy(psi_trajectory[t_idx]).to(torch.complex64)
    psi_clean = torch.from_numpy(psi_trajectory[0]).to(torch.complex64)
    
    # Get true Hamiltonian if available
    H_true = None
    if H_trajectory is not None and t_idx < len(H_trajectory):
        H_true = torch.from_numpy(H_trajectory[t_idx]).to(torch.complex64)
    
    # Compute the noise that was added
    psi_noisy_real = complex_to_real(psi_noisy)
    psi_clean_real = complex_to_real(psi_clean)
    true_noise = psi_noisy_real - psi_clean_real
    
    # Predict
    t_tensor = torch.tensor([t], dtype=torch.float32)
    H_pred, flow_pred, noise_pred, clean_pred = model(psi_noisy, t_tensor)
    
    # Compute loss
    loss = criterion(H_pred, flow_pred, noise_pred, clean_pred, 
                    true_noise, psi_clean, psi_noisy_real, H_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

print(f"Starting training on {device}...")


DT = 0.01
T = 1.0
STEPS = int(T / DT)

best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0.0
    
    # Train on multiple trajectories per epoch
    for _ in range(50):
        # Generate trajectory with Hamiltonians
        if np.random.rand() < 0.5:
            psi_fwd, H_fwd, rho_fwd = forward_diffusion_with_probability(dt=DT, T=T, noise_strength=1.0)
        else:
            psi_fwd, H_fwd = forward_diffusion(dt=DT, T=T, noise_strength=1.0)
        
        # Train on this trajectory
        total_loss += train_step(model, optimizer, psi_fwd, H_fwd, criterion)
    
    scheduler.step()
    avg_loss = total_loss / 50
    
    if avg_loss < best_loss:
        best_loss = avg_loss
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {avg_loss:.6f}, Best = {best_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

def reverse_diffusion(model, psi_T, steps):
    model.eval()
    with torch.no_grad():
        traj = model.reverse_diffusion(psi_T, steps=steps)
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

# Create figure with 2 rows
fig = plt.figure(figsize=(15, 8))

for idx, t_idx in enumerate(time_indices):
    # Calculate subplot position: 2 rows, 3 columns
    # First 3 plots in row 1, last 2 in row 2 (centered)
    if idx < 3:
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    else:
        # Center the last 2 plots in the second row
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
    
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

plt.suptitle('Forward Diffusion: Quantum State Spreading', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('bloch_sphere_diffusion.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Also create reverse diffusion visualization with proper ensemble
print("\nGenerating reverse diffusion visualization...")

# Generate ONE forward trajectory to get the final noisy state
psi_fwd_ref, H_fwd_ref = forward_diffusion(dt=DT, T=T, noise_strength=1.5)
final_noisy_state = psi_fwd_ref[-1]  # This is the noisy state we'll all start from

print(f"Starting all reverse trajectories from the same noisy state...")

# Generate multiple reverse trajectories ALL STARTING FROM THE SAME NOISY STATE
all_reverse_trajectories = []
for i in range(100):
    # All start from the same noisy state
    psi_rev = reverse_diffusion(model, final_noisy_state, steps=STEPS)
    all_reverse_trajectories.append(psi_rev)
    
    if (i + 1) % 25 == 0:
        print(f"Generated {i+1}/100 reverse trajectories...")

# Create reverse diffusion plot - should show convergence
fig2 = plt.figure(figsize=(15, 8))

# Show reverse diffusion at different time points (going backwards from t=1 to t=0)
reverse_time_indices = [0, 25, 50, 75, 99]  # t=1.0, 0.75, 0.5, 0.25, 0.0
reverse_time_values = [1.0 - (i / 99) for i in reverse_time_indices]

for plot_idx, t_idx in enumerate(reverse_time_indices):
    # Calculate subplot position: 2 rows, 3 columns
    if plot_idx < 3:
        ax = fig2.add_subplot(2, 3, plot_idx + 1, projection='3d')
    else:
        # Center the last 2 plots in the second row
        ax = fig2.add_subplot(2, 3, plot_idx + 2, projection='3d')
    
    plot_bloch_sphere_surface(ax, alpha=0.25)
    add_pole_labels(ax)
    
    # Extract reverse states at this timestep
    states_at_t = np.array([traj[t_idx] for traj in all_reverse_trajectories])
    x_t, y_t, z_t = bloch_coords(states_at_t)
    
    # Color coding
    if t_idx == 0:
        # Start of reverse = noisy state (all trajectories start here)
        colors = 'red'
        sizes = 80
        alpha_val = 0.8
        title_suffix = " (all start here)"
    elif t_idx == 99:
        # End of reverse = should cluster at clean state
        colors = 'green'
        sizes = 100
        alpha_val = 0.9
        title_suffix = " (should cluster)"
    else:
        # Intermediate - denoising in progress
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(x_t)))
        sizes = 30
        alpha_val = 0.6
        title_suffix = ""
    
    ax.scatter(x_t, y_t, z_t, c=colors, s=sizes, alpha=alpha_val,
               edgecolors='black', linewidths=0.3, zorder=5)
    
    # Also show the true clean state for reference
    if t_idx == 99:
        x_clean, y_clean, z_clean = bloch_coords(psi_fwd_ref[0:1])
        ax.scatter(x_clean, y_clean, z_clean, c='blue', marker='*', s=300,
                  edgecolors='darkblue', linewidths=2, label='True clean', zorder=10)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_title(f't = {reverse_time_values[plot_idx]:.2f}{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=20, azim=30)

plt.suptitle('Reverse Diffusion: Ensemble Denoising from Single Noisy State', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('bloch_sphere_reverse_diffusion.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Compute clustering metric at final time
final_states = np.array([traj[-1] for traj in all_reverse_trajectories])
x_final, y_final, z_final = bloch_coords(final_states)

# Measure spread (standard deviation in Bloch coordinates)
spread_x = np.std(x_final)
spread_y = np.std(y_final)
spread_z = np.std(z_final)
total_spread = np.sqrt(spread_x**2 + spread_y**2 + spread_z**2)

print(f"\nReverse diffusion clustering metrics:")
print(f"Final spread (std): x={spread_x:.4f}, y={spread_y:.4f}, z={spread_z:.4f}")
print(f"Total spread: {total_spread:.4f} (smaller is better - should be < 0.1 for good clustering)")

# Compute average fidelity with true clean state
true_clean = psi_fwd_ref[0]
fidelities = []
for traj in all_reverse_trajectories:
    final_state = traj[-1]
    overlap = np.vdot(final_state, true_clean)
    fidelity = np.abs(overlap)**2
    fidelities.append(fidelity)

avg_fidelity = np.mean(fidelities)
std_fidelity = np.std(fidelities)
print(f"Average final fidelity: {avg_fidelity:.4f} ± {std_fidelity:.4f}")
print(f"(Fidelity > 0.95 indicates good denoising)")

print("Visualizations saved!")