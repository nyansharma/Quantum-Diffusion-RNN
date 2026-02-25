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
    norm = torch.sqrt(torch.sum(c.real**2 + c.imag**2, dim=-1, keepdim=True))
    return c / (norm + 1e-8)
def fokker_planck_update(rho, psi, O, dt):
    f_psi = drift_vector(psi, O)
    
    g_psi = noise_vector(O, psi)
    
    eps = 1e-6
    drift_contrib = 0.0
    
    for i in range(2):
        psi_plus = psi.copy()
        psi_plus[i] += eps
        psi_plus = psi_plus / np.linalg.norm(psi_plus)
        
        f_plus = drift_vector(psi_plus, O)
        drift_contrib -= np.real((f_plus[i] - f_psi[i]) / eps) * rho

    diffusion_contrib = 0.0
    
    for i in range(2):
        for j in range(2):
            
            psi_plus = psi.copy()
            psi_minus = psi.copy()
            psi_plus[i] += eps
            psi_minus[i] -= eps
            
            psi_plus = psi_plus / np.linalg.norm(psi_plus)
            psi_minus = psi_minus / np.linalg.norm(psi_minus)
            
         
            D_ij = np.real(np.conj(g_psi[i]) * g_psi[j])
            
            
            if i == j:
              
                d2_rho = (rho - rho) / (eps * eps)  
            
            diffusion_contrib += 0.5 * D_ij * d2_rho
    
    
    drho_dt = drift_contrib + diffusion_contrib
    
    return rho + drho_dt * dt

def forward_diffusion_with_probability(dt, T, noise_strength=0.5):

    n = int(T / dt)
    psi = np.zeros((n, 2), dtype=np.complex128)
    psi[0] = psi_0.astype(np.complex128)
    rho = np.ones(n, dtype=np.float64)  
    rho[0] = 1.0  
    H = np.zeros((n, 2, 2), dtype=np.complex128)
    
    for i in range(n - 1):
        
        O = PAULI[np.random.randint(0, 4)]
        
        
        drift = drift_vector(psi[i], O)
        
        
        dW_real = np.random.randn(2) * np.sqrt(dt)
        dW_imag = np.random.randn(2) * np.sqrt(dt)
        dW = dW_real + 1j * dW_imag
        
        
        noise = noise_vector(O, psi[i])
        
        
        current_noise_strength = noise_strength * (i / n)
        
    
        delta_psi = drift * dt + current_noise_strength * noise * np.linalg.norm(dW)
        psi[i+1] = psi[i] + delta_psi
        psi[i+1] = psi[i+1] / np.linalg.norm(psi[i+1])
        
    
        rho[i+1] = fokker_planck_update(rho[i], psi[i], O, dt)
        
        rho[i+1] = max(rho[i+1], 1e-10)
        
        H[i] = Hamiltonian(drift, psi[i])
    
    rho = rho / np.sum(rho)
    
    return psi, H, rho


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
def forward_diffusion(dt, T, noise_strength=0.5):
    n = int(T / dt)
    psi = np.zeros((n, 2), dtype=np.complex128)
    psi[0] = psi_0.astype(np.complex128)
    H = np.zeros((n, 2, 2), dtype=np.complex128)
    
    for i in range(n - 1):
    
        O = PAULI[np.random.randint(0, 4)]
        
        drift = drift_vector(psi[i], O)
        
    
        dW_real = np.random.randn(2) * np.sqrt(dt)
        dW_imag = np.random.randn(2) * np.sqrt(dt)
        dW = dW_real + 1j * dW_imag
        
       
        noise = noise_vector(O, psi[i])
        
        current_noise_strength = noise_strength * (i / n)
        
    
        delta_psi = drift * dt + current_noise_strength * noise * np.linalg.norm(dW)
        
        psi[i+1] = psi[i] + delta_psi
        
        psi[i+1] = psi[i+1] / np.linalg.norm(psi[i+1])
        
        flow = drift  
        H[i] = Hamiltonian(flow, psi[i])
    
    return psi, H

class HamiltonianDenoisingNetwork(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
        self.hamiltonian_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 8)  
        )
        
     
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)  
        )
        
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)
        )
        
       
        self.clean_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, psi_noisy, t):
       
        if isinstance(psi_noisy, np.ndarray):
            psi_noisy = torch.from_numpy(psi_noisy)
        
        if torch.is_complex(psi_noisy):
            psi_real = complex_to_real(psi_noisy.to(torch.complex64))
        else:
            psi_real = psi_noisy
        
      
        if isinstance(t, (float, int)):
            t = torch.tensor([t], dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if psi_real.dim() == 1:
            t = t.expand(1)
        
        
        x = torch.cat([psi_real, t], dim=-1)
        
        
        features = self.encoder(x)
        
        
        H_flat = self.hamiltonian_predictor(features) 
        H_real = H_flat[:4].reshape(2, 2)
        H_imag = H_flat[4:].reshape(2, 2)
        
        H_real = 0.5 * (H_real + H_real.T)  
        H_imag = 0.5 * (H_imag - H_imag.T)  
        H_complex = torch.complex(H_real, H_imag)
        
        flow_pred = self.flow_predictor(features)
   
        noise_pred = self.noise_predictor(features)
       
        clean_pred = self.clean_predictor(features)
        
        clean_complex = real_to_complex(clean_pred)
        clean_normalized = complex_to_real(clean_complex)
        
        return H_complex, flow_pred, noise_pred, clean_normalized
    
    def hamiltonian_evolution(self, psi, H, dt):

        if not torch.is_complex(psi):
            psi = real_to_complex(psi)
        

        evolution = torch.eye(2, dtype=torch.complex64) - 1j * H * dt
        
        
        psi_evolved = evolution @ psi
        
        
        psi_evolved = psi_evolved / torch.sqrt(torch.sum(torch.abs(psi_evolved)**2))
        
        return psi_evolved
    
    def denoise_step_with_hamiltonian(self, psi_noisy, t, dt=0.01, use_hamiltonian=True):
       
        H_pred, flow_pred, noise_pred, clean_pred = self.forward(psi_noisy, t)
        
        if torch.is_complex(psi_noisy):
            psi_real = complex_to_real(psi_noisy.to(torch.complex64))
        else:
            psi_real = psi_noisy
        
        
        if use_hamiltonian and t > 0.3:  
            psi_complex = real_to_complex(psi_real)
            psi_evolved = self.hamiltonian_evolution(psi_complex, H_pred, -dt)  
            denoised = complex_to_real(psi_evolved)
        
        elif t > 0.1:
            alpha = 1 - t
            denoised = psi_real + flow_pred * dt - (1 - alpha) * noise_pred * dt
        
    
        else:
            
            denoised = 0.7 * clean_pred + 0.3 * psi_real
        
        
        denoised_complex = real_to_complex(denoised)
        return complex_to_real(denoised_complex)
    
    def reverse_diffusion(self, psi_T, steps, use_hamiltonian=True):
      
        if isinstance(psi_T, np.ndarray):
            psi_T = torch.from_numpy(psi_T)
        
        psi_T = psi_T.to(torch.complex64)
        curr = complex_to_real(psi_T)
        
        trajectory = [curr]
        dt = 1.0 / steps
        
       
        for i in range(steps):
            t = torch.tensor([1.0 - (i + 1) / steps], dtype=torch.float32)
            
            curr = self.denoise_step_with_hamiltonian(curr, t, dt, use_hamiltonian)
            
            trajectory.append(curr)
        
        return torch.stack(trajectory)


class HamiltonianDenoisingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, H_pred, flow_pred, noise_pred, clean_pred, 
                true_noise, true_clean, psi_current, H_true=None):

        true_clean = true_clean.to(torch.complex64)
        true_clean_real = complex_to_real(true_clean)
        
    
        clean_pred_complex = real_to_complex(clean_pred)
        overlap = torch.sum(clean_pred_complex.conj() * true_clean)
        fidelity = torch.abs(overlap) ** 2
        clean_loss = 1 - fidelity
        
        if true_noise is not None:
            noise_loss = torch.mean((noise_pred - true_noise) ** 2)
        else:
            noise_loss = 0.0
       
        H_herm_error = torch.sum(torch.abs(H_pred - H_pred.conj().T) ** 2)
        
        
        psi_complex = real_to_complex(true_clean_real)
        energy = torch.real(psi_complex.conj() @ H_pred @ psi_complex)
        energy_penalty = torch.relu(torch.abs(energy) - 10.0)  
        
        hamiltonian_loss = H_herm_error + 0.1 * energy_penalty

        denoising_direction = true_clean_real - complex_to_real(real_to_complex(true_clean_real))
        flow_alignment = -torch.sum(flow_pred * denoising_direction)  
        
        
        total_loss = (
            0.5 * clean_loss +           
            0.2 * noise_loss +            
            0.1 * hamiltonian_loss +      
            0.05 * flow_alignment         
        )
        
        return total_loss

model = HamiltonianDenoisingNetwork(hidden_dim=256)
EPOCHS = 100
criterion = HamiltonianDenoisingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train_step(model, optimizer, psi_trajectory, H_trajectory, criterion):
    optimizer.zero_grad()
    
    
    n_steps = len(psi_trajectory)
    t_idx = np.random.randint(1, n_steps)
    t = t_idx / n_steps
    
    
    psi_noisy = torch.from_numpy(psi_trajectory[t_idx]).to(torch.complex64)
    psi_clean = torch.from_numpy(psi_trajectory[0]).to(torch.complex64)
    
    
    H_true = None
    if H_trajectory is not None and t_idx < len(H_trajectory):
        H_true = torch.from_numpy(H_trajectory[t_idx]).to(torch.complex64)
    
    psi_noisy_real = complex_to_real(psi_noisy)
    psi_clean_real = complex_to_real(psi_clean)
    true_noise = psi_noisy_real - psi_clean_real
    

    t_tensor = torch.tensor([t], dtype=torch.float32)
    H_pred, flow_pred, noise_pred, clean_pred = model(psi_noisy, t_tensor)
    
  
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
    

    for _ in range(50):
        
        if np.random.rand() < 0.5:
            psi_fwd, H_fwd, rho_fwd = forward_diffusion_with_probability(dt=DT, T=T, noise_strength=1.0)
        else:
            psi_fwd, H_fwd = forward_diffusion(dt=DT, T=T, noise_strength=1.0)
        
    
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

final_overlap = np.vdot(psi_rev[-1], psi_fwd[0])
final_fidelity = np.abs(final_overlap)**2
print(f"\nFinal fidelity: {final_fidelity:.6f}")

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_sphere_surface(ax, alpha=0.3):

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=alpha, color='lightgray', 
                    linewidth=0, antialiased=True, shade=True, zorder=1)

def add_pole_labels(ax):

    ax.scatter([0], [0], [1], color='black', s=80, alpha=0.7, zorder=10)
    ax.scatter([0], [0], [-1], color='black', s=80, alpha=0.7, zorder=10)
    ax.text(-0.15, 0, 1.15, '|0⟩', fontsize=14, fontweight='bold', ha='center')
    ax.text(-0.15, 0, -1.15, '|1⟩', fontsize=14, fontweight='bold', ha='center')


np.random.seed(42)
n_trajectories = 500

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


time_indices = [0, 3, 10, 30, 99]  
time_values = [i * DT for i in time_indices]


fig = plt.figure(figsize=(15, 8))

for idx, t_idx in enumerate(time_indices):

    if idx < 3:
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    else:
        
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
    
 
    plot_bloch_sphere_surface(ax, alpha=0.25)
    add_pole_labels(ax)
   
    states_at_t = np.array([traj[t_idx] for traj in all_trajectories])
    probs_at_t = np.array([prob[t_idx] for prob in all_probabilities])
    
    x_t, y_t, z_t = bloch_coords(states_at_t)
    

    if t_idx == 0:
        
        colors = 'red'
        sizes = 100
        alpha = 1.0
    else:
    
        prob_normalized = probs_at_t / (np.max(probs_at_t) + 1e-10)
        colors = plt.cm.Reds(0.3 + 0.6 * prob_normalized)
        sizes = 10 + 40 * prob_normalized  
        alpha = 0.6
    
    
    ax.scatter(x_t, y_t, z_t, c=colors, s=sizes, alpha=alpha, 
               edgecolors='darkred', linewidths=0.3, zorder=5)
    
   
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_title(f't = {time_values[idx]:.2f}', fontsize=16, fontweight='bold', pad=10)
    

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.view_init(elev=20, azim=30)

plt.suptitle('Forward Diffusion: Quantum State Spreading', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('bloch_sphere_diffusion.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\nGenerating reverse diffusion visualization...")

psi_fwd_ref, H_fwd_ref = forward_diffusion(dt=DT, T=T, noise_strength=1.5)
final_noisy_state = psi_fwd_ref[-1]  

print(f"Starting all reverse trajectories from the same noisy state...")

all_reverse_trajectories = []
for i in range(100):
    psi_rev = reverse_diffusion(model, final_noisy_state, steps=STEPS)
    all_reverse_trajectories.append(psi_rev)
    
    if (i + 1) % 25 == 0:
        print(f"Generated {i+1}/100 reverse trajectories...")

fig2 = plt.figure(figsize=(15, 8))

reverse_time_indices = [0, 25, 50, 75, 99]  
reverse_time_values = [1.0 - (i / 99) for i in reverse_time_indices]

for plot_idx, t_idx in enumerate(reverse_time_indices):
 
    if plot_idx < 3:
        ax = fig2.add_subplot(2, 3, plot_idx + 1, projection='3d')
    else:
       
        ax = fig2.add_subplot(2, 3, plot_idx + 2, projection='3d')
    
    plot_bloch_sphere_surface(ax, alpha=0.25)
    add_pole_labels(ax)
    
    
    states_at_t = np.array([traj[t_idx] for traj in all_reverse_trajectories])
    x_t, y_t, z_t = bloch_coords(states_at_t)

    if t_idx == 0:
        colors = 'red'
        sizes = 80
        alpha_val = 0.8
        title_suffix = " (all start here)"
    elif t_idx == 99:
        colors = 'green'
        sizes = 100
        alpha_val = 0.9
        title_suffix = " (should cluster)"
    else:
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(x_t)))
        sizes = 30
        alpha_val = 0.6
        title_suffix = ""
    
    ax.scatter(x_t, y_t, z_t, c=colors, s=sizes, alpha=alpha_val,
               edgecolors='black', linewidths=0.3, zorder=5)
    
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

final_states = np.array([traj[-1] for traj in all_reverse_trajectories])
x_final, y_final, z_final = bloch_coords(final_states)

spread_x = np.std(x_final)
spread_y = np.std(y_final)
spread_z = np.std(z_final)
total_spread = np.sqrt(spread_x**2 + spread_y**2 + spread_z**2)

print(f"\nReverse diffusion clustering metrics:")
print(f"Final spread (std): x={spread_x:.4f}, y={spread_y:.4f}, z={spread_z:.4f}")
print(f"Total spread: {total_spread:.4f} (smaller is better - should be < 0.1 for good clustering)")

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