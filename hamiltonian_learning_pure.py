import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")


X  = np.array([[0, 1],  [1, 0]],   dtype=complex)
Y  = np.array([[0,-1j], [1j, 0]],  dtype=complex)
Z  = np.array([[1, 0],  [0, -1]],  dtype=complex)
I2 = np.eye(2,                      dtype=complex)
PAULI_XYZ = [X, Y, Z]   

def complex_to_real(psi: torch.Tensor) -> torch.Tensor:
    return torch.cat([psi.real, psi.imag], dim=-1)

def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    re, im = x[..., :2], x[..., 2:]
    c      = torch.complex(re, im)
    norm   = (c.real**2 + c.imag**2).sum(-1, keepdim=True).sqrt()
    return c / (norm + 1e-8)

def bloch_coords(psi: np.ndarray) -> np.ndarray:
  
    if psi.ndim == 1:
        psi = psi[None]
    bx = 2 * np.real(np.conj(psi[:, 0]) * psi[:, 1])
    by = 2 * np.imag(np.conj(psi[:, 0]) * psi[:, 1]) 
    bz = np.abs(psi[:, 0])**2 - np.abs(psi[:, 1])**2
    return np.stack([bx, by, bz], axis=-1)              
def bloch_to_qubit(b: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(b)
    if r < 1e-8:
        return np.array([1., 0.], dtype=complex)
    b     = b / r
    theta = np.arccos(np.clip(b[2], -1, 1))
    phi   = np.arctan2(b[1], b[0])
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)

def random_qubit(rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    return (z / np.linalg.norm(z)).astype(complex)

def forward_diffusion(psi_0: np.ndarray,
                      n_steps: int = 200,
                      noise_strength: float = 1.5,
                      rng: np.random.Generator = None) -> dict:

    if rng is None:
        rng = np.random.default_rng()
    dt  = 1.0 / n_steps
    psi = np.zeros((n_steps, 2), dtype=np.complex128)
    H   = np.zeros((n_steps, 2, 2), dtype=np.complex128)
    psi[0] = psi_0.copy()

    for i in range(n_steps - 1):
        O = PAULI_XYZ[i % 3]         
        dW  = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) * np.sqrt(dt)
        dW -= np.dot(psi[i].conj(), dW) * psi[i]

        O_psi  = O @ psi[i]
        exp_O  = np.vdot(psi[i], O_psi)
        tan_d  = O_psi - exp_O * psi[i]

        psi[i+1] = psi[i] + 0.1 * tan_d * dt + noise_strength * dW
        psi[i+1] /= np.linalg.norm(psi[i+1])

        h = 1j * np.outer(tan_d, psi[i].conj())
        H[i] = h + h.conj().T

    H[-1] = H[-2]  
    return {
        "psi":   psi,
        "bloch": bloch_coords(psi),
        "H":     H,
        "times": np.linspace(0.0, 1.0, n_steps),
    }


class BlochTokenizer:

    def __init__(self, K: int = 256):
        self.K         = K
        self.centroids = self._fibonacci_sphere(K)   # (K,3)

    @staticmethod
    def _fibonacci_sphere(K: int) -> np.ndarray:
        golden = (1 + np.sqrt(5)) / 2
        i      = np.arange(K, dtype=float)
        theta  = np.arccos(1 - 2 * (i + 0.5) / K)
        phi    = 2 * np.pi * i / golden
        return np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=-1).astype(np.float32)

    def encode(self, bloch: np.ndarray) -> np.ndarray:
        return (bloch.astype(np.float32) @ self.centroids.T).argmax(axis=-1)

    def soft_encode(self, bloch: np.ndarray, top_k: int = 8) -> tuple:
        sims = bloch.astype(np.float32) @ self.centroids.T   # (N,K)
        idx  = np.argsort(sims, axis=-1)[:, -top_k:][:, ::-1].copy()
        sim  = np.take_along_axis(sims, idx, axis=-1)
        exp  = np.exp(sim - sim.max(-1, keepdims=True))
        w    = exp / exp.sum(-1, keepdims=True)
        return idx, w


class BlochEmbedding(nn.Module):
  
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)

    def forward_soft(self, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
       
        return (self.embed(idx) * w.unsqueeze(-1)).sum(0)

    def forward_batch(self, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    
        return (self.embed(idx) * w.unsqueeze(-1)).sum(1)



class QuantumDiffusionRNN(nn.Module):
   
    def __init__(self,
                 num_tokens: int  = 256,
                 embed_dim: int   = 48,
                 hidden_dim: int  = 256,
                 n_layers: int    = 2,
                 top_k: int       = 8,
                 dropout: float   = 0.1):
        super().__init__()
        self.tokenizer  = BlochTokenizer(K=num_tokens)
        self.embedding  = BlochEmbedding(num_tokens, embed_dim)
        self.top_k      = top_k
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        input_dim = 3 + 3 + embed_dim

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        self.bloch_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 3),
        )

    def _step_input(self, bloch: np.ndarray, t: float) -> torch.Tensor:
        b   = torch.tensor(bloch, dtype=torch.float32)        
        tv  = torch.tensor([t],   dtype=torch.float32)
        tf  = torch.cat([tv, torch.sin(np.pi*tv), torch.cos(np.pi*tv)])  
        bidx, bw = self.tokenizer.soft_encode(bloch[None], top_k=self.top_k)
        idx_t    = torch.from_numpy(bidx[0]).long()
        w_t      = torch.from_numpy(bw[0].astype(np.float32))
        tok_e    = self.embedding.forward_soft(idx_t, w_t)     
        return torch.cat([b, tf, tok_e])                    

    def _seq_inputs(self, bloch_seq: np.ndarray,
                    t_seq: np.ndarray) -> torch.Tensor:
        return torch.stack([
            self._step_input(bloch_seq[i], float(t_seq[i]))
            for i in range(len(t_seq))
        ])

    @staticmethod
    def _unit(raw: torch.Tensor) -> torch.Tensor:
        return raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _flow_to_H(flow_real: torch.Tensor,
                   psi_c: torch.Tensor) -> torch.Tensor:
        v = real_to_complex(flow_real.unsqueeze(0)).squeeze(0)
        H = 1j * torch.outer(v, psi_c.conj())
        return H + H.conj().T

    def forward_sequence(self,
                         bloch_seq: np.ndarray,
                         t_seq: np.ndarray,
                         hidden=None) -> tuple:
  
        x   = self._seq_inputs(bloch_seq, t_seq).unsqueeze(0)  # (1,L,D)
        out, hidden = self.lstm(x, hidden)                       # (1,L,H)
        b0_raw   = self.bloch_head(out.squeeze(0))               # (L,3)
        b0_preds = self._unit(b0_raw)                            # (L,3)
        return b0_preds, hidden

    def step(self,
             bloch: np.ndarray,
             t: float,
             hidden=None) -> tuple:
     
        x   = self._step_input(bloch, t).unsqueeze(0).unsqueeze(0)  # (1,1,D)
        out, hidden = self.lstm(x, hidden)                            # (1,1,H)
        out = out.squeeze(0).squeeze(0)                               # (H,)

        b0_raw  = self.bloch_head(out)
        b0_pred = self._unit(b0_raw)                                  # (3,)

        return b0_pred, hidden

    def reverse_diffusion(self,
                          psi_T: np.ndarray,
                          n_steps: int = 50,
                          noise_scale: float = 0.08) -> dict:
   
        self.eval()
        dt     = 1.0 / n_steps
        psi    = psi_T.astype(np.complex128).copy()
        hidden = None

        traj, H_traj, x0_traj, t_vals = [psi.copy()], [], [], []

        with torch.no_grad():
            for i in range(n_steps):
                t_val = max(1.0 - i * dt, dt)

                bloch_curr = bloch_coords(psi[None])[0]     

                b0_pred, hidden = self.step(bloch_curr, t_val, hidden)

                psi0_np = bloch_to_qubit(b0_pred.numpy())   

                psi_tc  = torch.from_numpy(psi).to(torch.complex64)
                psi0_tc = torch.from_numpy(psi0_np).to(torch.complex64)
                flow_c  = (psi0_tc - psi_tc) / max(t_val, 0.01)
                flow_r  = complex_to_real(flow_c.unsqueeze(0)).squeeze(0)
                H_pred  = self._flow_to_H(flow_r, psi_tc)

                alpha   = min(dt / t_val, 1.0)
                psi_new = (1 - alpha) * psi_tc + alpha * psi0_tc
                psi_new = psi_new / psi_new.abs().pow(2).sum().sqrt()

                if noise_scale > 0 and t_val > 2 * dt:
                    psi_np = psi_new.numpy().astype(np.complex128)
                    O      = PAULI_XYZ[i % 3]
                    g_psi  = (O - np.vdot(psi_np, O @ psi_np) * I2) @ psi_np
                    g_norm = np.linalg.norm(g_psi)
                    if g_norm > 1e-8:
                        xi   = g_psi / g_norm * np.random.randn()
                        xi  *= noise_scale * np.sqrt(dt) * t_val
                        xi_c = torch.tensor(
                            [xi.real, xi.imag], dtype=torch.float32).flatten()
                        xi_c = torch.complex(xi_c[:2], xi_c[2:]).to(torch.complex64)
                        xi_c -= (psi_new.conj() * xi_c).sum() * psi_new  # tangent project
                        psi_new = psi_new + xi_c
                        psi_new = psi_new / psi_new.abs().pow(2).sum().sqrt()

                psi = psi_new.numpy().astype(np.complex128)

                traj.append(psi.copy())
                H_traj.append(H_pred.numpy())
                x0_traj.append(psi0_np.copy())
                t_vals.append(t_val)

        return {
            "trajectory":   np.array(traj),    
            "hamiltonians": np.array(H_traj),     
            "x0_preds":     np.array(x0_traj),  
            "times":        np.array(t_vals),     
        }


def train(model: QuantumDiffusionRNN,
          psi_0: np.ndarray,
          n_epochs: int        = 300,
          n_traj_per_epoch: int = 20,
          sde_steps: int       = 200,
          seq_len: int         = 50,
          noise_strength: float = 1.5,
          lr: float            = 3e-4) -> list:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def lr_lambda(ep):
        warmup = 30
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(n_epochs - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    b0_true = torch.tensor(bloch_coords(psi_0[None])[0], dtype=torch.float32)  # (3,)
    rng     = np.random.default_rng(seed=0)
    losses  = []
    best    = float('inf')

    print(f"Training LSTM on {device} | {n_epochs} epochs × {n_traj_per_epoch} traj")
    print(f"Fixed |ψ₀⟩ Bloch = {b0_true.numpy().round(3)}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(n_traj_per_epoch):
            optimizer.zero_grad()

            traj = forward_diffusion(psi_0, n_steps=sde_steps,
                                     noise_strength=noise_strength, rng=rng)
           
            all_idx  = np.arange(1, sde_steps)
            sub_idx  = np.round(
                np.linspace(0, len(all_idx)-1, seq_len)).astype(int)
            sub_idx  = all_idx[sub_idx]

            bloch_s  = traj["bloch"][sub_idx]   
            t_s      = traj["times"][sub_idx]   

           
            bloch_r  = bloch_s[::-1].copy()
            t_r      = t_s[::-1].copy()

            b0_preds, _ = model.forward_sequence(bloch_r, t_r)  

          
            w = torch.tensor(1.0 - t_r, dtype=torch.float32) 
            w = w / w.sum()

            b0_t  = b0_true.unsqueeze(0).expand_as(b0_preds)   
            dot   = (b0_preds * b0_t).sum(dim=-1)               
            loss  = (0.5 * (1.0 - dot) * w).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / n_traj_per_epoch
        losses.append(avg)
        if avg < best:
            best = avg

        if (epoch + 1) % 50 == 0:
            model.eval()
            fids = []
            with torch.no_grad():
                for _ in range(15):
                    t_val  = forward_diffusion(psi_0, n_steps=200, rng=rng)
                    step_i = int(0.8 * 199)
                    b_curr = t_val["bloch"][step_i]
                    b0p, _ = model.step(b_curr, 0.8, hidden=None)
                    fid    = 0.5 * (1 + float(
                        (b0p.detach().numpy() * b0_true.numpy()).sum()))
                    fids.append(fid)
            model.train()
            print(f"Ep {epoch+1:3d}/{n_epochs} | "
                  f"loss={avg:.5f}  best={best:.5f} | "
                  f"fid@t=0.8: {np.mean(fids):.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return losses


def evaluate(model: QuantumDiffusionRNN,
             psi_0: np.ndarray,
             n_trials: int  = 50,
             n_steps: int   = 50) -> dict:

    model.eval()
    fidelities = []
    rng = np.random.default_rng(seed=999)

    for trial in range(n_trials):
        traj   = forward_diffusion(psi_0, n_steps=200, noise_strength=1.5, rng=rng)
        psi_T  = traj["psi"][-1]

        rev    = model.reverse_diffusion(psi_T, n_steps=n_steps, noise_scale=0.0)
        final  = rev["trajectory"][-1]
        fid    = abs(np.vdot(final, psi_0))**2
        fidelities.append(fid)

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1:2d}/{n_trials}  "
                  f"running mean = {np.mean(fidelities):.4f}")

    fa = np.array(fidelities)

    traj_r   = forward_diffusion(psi_0, n_steps=200, noise_strength=1.5,
                                 rng=np.random.default_rng(seed=42))
    psi_T_r  = traj_r["psi"][-1]
    finals   = []
    for _ in range(20):
        rev = model.reverse_diffusion(psi_T_r, n_steps=n_steps, noise_scale=0.08)
        finals.append(rev["trajectory"][-1])
    bvecs  = bloch_coords(np.array(finals))
    spread = float(np.sqrt(bvecs[:,0].std()**2 + bvecs[:,1].std()**2 + bvecs[:,2].std()**2))

    results = {
        "mean_fidelity": float(fa.mean()),
        "std_fidelity":  float(fa.std()),
        "min_fidelity":  float(fa.min()),
        "max_fidelity":  float(fa.max()),
        "bloch_spread":  spread,
    }
    print(f"\nFidelity : {results['mean_fidelity']:.4f} ± {results['std_fidelity']:.4f}")
    print(f"Spread   : {results['bloch_spread']:.4f}  (target < 0.1)")
    return results



CMAP_FWD = LinearSegmentedColormap.from_list(
    "fwd", ["#0a1628", "#1a3a6e", "#2060c0", "#40a0ff", "#80d4ff"])
CMAP_REV = LinearSegmentedColormap.from_list(
    "rev", ["#1a0a00", "#5c2a00", "#c06010", "#f0a830", "#ffe090"])


def _sphere(ax, alpha=0.10):
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi,   60)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)),
                    alpha=alpha, color="#1a2a3a", linewidth=0,
                    antialiased=True, shade=False, zorder=0)


def _circles(ax):
    t = np.linspace(0, 2*np.pi, 120)
    for a, b, c in [(np.cos(t), np.sin(t), np.zeros_like(t)),
                    (np.cos(t), np.zeros_like(t), np.sin(t)),
                    (np.zeros_like(t), np.cos(t), np.sin(t))]:
        ax.plot(a, b, c, lw=0.4, color="#243040", alpha=0.5, zorder=1)


def _poles(ax):
    kw = dict(color="#e8e8f0", fontsize=9, fontweight="bold",
              ha="center", va="center", zorder=20)
    ax.text(0, 0,  1.22, "|0⟩", **kw)
    ax.text(0, 0, -1.22, "|1⟩", **kw)
    ax.scatter([0], [0], [ 1], c="#e8e8f0", s=18, zorder=15, depthshade=False)
    ax.scatter([0], [0], [-1], c="#e8e8f0", s=18, zorder=15, depthshade=False)


def _style(ax, title="", tc="#c0d8f8"):
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), zlim=(-1.3, 1.3))
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor("none")
    ax.set_facecolor("#050d1a")
    if title:
        ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=4)
    ax.view_init(elev=18, azim=35)


def _target_dot(ax, psi_0):
    b = bloch_coords(psi_0[None])[0]
    ax.scatter([b[0]], [b[1]], [b[2]], c="#ff4466", s=200, zorder=30,
               edgecolors="white", linewidths=1.6, depthshade=False)


def make_panels(psi_0, fwd_trajs, rev_trajs, out_path):
    N_FWD = len(fwd_trajs[0])
    N_REV = len(rev_trajs[0])
    FWD_F = [0.00, 0.08, 0.20, 0.45, 1.00]
    REV_F = [1.00, 0.75, 0.50, 0.25, 0.00]

    fig = plt.figure(figsize=(18, 8), facecolor="#060e1c")
    fig.text(0.5, 0.97, "QUANTUM STATE DIFFUSION ON THE BLOCH SPHERE",
             ha="center", va="top", color="#c8dcf4", fontsize=15,
             fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.935,
             "Forward: pure-state SDE spreads |ψ⟩ over S²   ·   "
             "Reverse: LSTM Hamiltonian denoising recovers |ψ₀⟩  "
             f"(60 trajectories each)",
             ha="center", va="top", color="#5a7090", fontsize=9,
             fontfamily="monospace")

    def panel(row, col):
        return fig.add_subplot(2, 5, row*5+col+1,
                               projection="3d", facecolor="#060e1c")

    def pts(trajs, step):
        arr = np.array([bloch_coords(t[step][None])[0] for t in trajs])
        return arr[:, 0], arr[:, 1], arr[:, 2]

    for col, frac in enumerate(FWD_F):
        ax   = panel(0, col)
        step = int(frac * (N_FWD - 1))
        _sphere(ax); _circles(ax); _poles(ax)
        bx, by, bz = pts(fwd_trajs, step)
        normed = (bz + 1) / 2
        ax.scatter(bx, by, bz, c=CMAP_FWD(normed), s=20+30*normed,
                   alpha=0.85, depthshade=True, zorder=10, edgecolors="none")
        _target_dot(ax, psi_0)
        _style(ax, f"t = {frac:.2f}", "#5090d0")
        if col == 0:
            ax.text2D(-0.15, 0.5, "FORWARD\nDIFFUSION",
                      transform=ax.transAxes, rotation=90, ha="center",
                      va="center", color="#4080c0", fontsize=9,
                      fontweight="bold", fontfamily="monospace")

    for col, frac in enumerate(REV_F):
        ax   = panel(1, col)
        step = int((1.0 - frac) * (N_REV - 1))
        _sphere(ax); _circles(ax); _poles(ax)
        bx, by, bz = pts(rev_trajs, step)
        normed = (bz + 1) / 2
        ax.scatter(bx, by, bz, c=CMAP_REV(normed), s=20+30*normed,
                   alpha=0.85, depthshade=True, zorder=10, edgecolors="none")
        if frac < 0.05:       
            ax.scatter(bx, by, bz, c="#ffe090", s=55, alpha=0.9,
                      depthshade=False, zorder=15, edgecolors="none")
        _target_dot(ax, psi_0)
        _style(ax, f"t = {frac:.2f}", "#c09040")
        if col == 0:
            ax.text2D(-0.15, 0.5, "REVERSE\nDIFFUSION",
                      transform=ax.transAxes, rotation=90, ha="center",
                      va="center", color="#c09040", fontsize=9,
                      fontweight="bold", fontfamily="monospace")

    leg = fig.legend(handles=[
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#ff4466',
               markersize=10, label='|ψ₀⟩  fixed reference state',
               markeredgecolor='white', markeredgewidth=1.2),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#40a0ff',
               markersize=8, label='forward SDE ensemble (60 trajectories)',
               markeredgecolor='none'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#f0a830',
               markersize=8, label='reverse diffusion — LSTM Hamiltonian',
               markeredgecolor='none'),
    ], loc="lower center", ncol=3, framealpha=0.15, facecolor="#0a1628",
       edgecolor="#1a2a3a", labelcolor="#a0b8d0", fontsize=9,
       bbox_to_anchor=(0.5, 0.01))
    for txt in leg.get_texts():
        txt.set_fontfamily("monospace")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#060e1c")
    plt.close(fig)
    print(f"  Saved panels → {out_path}")


def make_animation(psi_0, fwd_trajs, rev_trajs, out_path, fps=20):
    N_FWD   = len(fwd_trajs[0])
    N_REV   = len(rev_trajs[0])
    N_TRAJ  = len(fwd_trajs)
    HOLD    = 18
    FWD_IDX = list(range(0, N_FWD, max(1, N_FWD//120))) + [N_FWD-1]
    REV_IDX = list(range(0, N_REV, max(1, N_REV//120))) + [N_REV-1]
    TOTAL   = len(FWD_IDX) + HOLD + len(REV_IDX) + HOLD

    fig = plt.figure(figsize=(7, 7), facecolor="#050d1a")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#050d1a")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _sphere(ax); _circles(ax); _poles(ax); _style(ax)

    ph_txt = ax.text2D(0.5, 0.96, "", transform=ax.transAxes, ha="center",
                       va="top", color="#c0d8f8", fontsize=13,
                       fontweight="bold", fontfamily="monospace")
    t_txt  = ax.text2D(0.5, 0.91, "", transform=ax.transAxes, ha="center",
                       va="top", color="#7090b0", fontsize=10,
                       fontfamily="monospace")

    sc_fwd = ax.scatter([], [], [], s=14, c="#2060c0", alpha=0.0, depthshade=True)
    sc_rev = ax.scatter([], [], [], s=14, c="#f0a830", alpha=0.0, depthshade=True)
    b0     = bloch_coords(psi_0[None])[0]
    ax.scatter([b0[0]], [b0[1]], [b0[2]], c="#ff4466", s=220, zorder=40,
               edgecolors="white", linewidths=2, depthshade=False)

    def update(frame):
        if frame < len(FWD_IDX):
            phase = "fwd"; idx = FWD_IDX[frame]; frac = idx/(N_FWD-1)
        elif frame < len(FWD_IDX) + HOLD:
            phase = "fwd"; idx = N_FWD-1; frac = 1.0
        elif frame < len(FWD_IDX) + HOLD + len(REV_IDX):
            phase = "rev"
            ridx  = frame - len(FWD_IDX) - HOLD
            idx   = REV_IDX[ridx]; frac = 1.0 - idx/(N_REV-1)
        else:
            phase = "rev"; idx = N_REV-1; frac = 0.0

        if phase == "fwd":
            sc_fwd.set_alpha(0.85); sc_rev.set_alpha(0.0)
            ph_txt.set_text("▶  FORWARD DIFFUSION"); ph_txt.set_color("#4090ff")
            t_txt.set_text(f"t = {frac:.3f}   (weak measurement noise)")
            pts = np.array([bloch_coords(t[idx][None])[0] for t in fwd_trajs])
            sc_fwd._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        else:
            sc_fwd.set_alpha(0.0); sc_rev.set_alpha(0.85)
            ph_txt.set_text("◀  REVERSE  (LSTM Hamiltonian)")
            ph_txt.set_color("#f0a830")
            t_txt.set_text(f"t = {frac:.3f}   (denoising toward |ψ₀⟩)")
            pts = np.array([bloch_coords(t[idx][None])[0] for t in rev_trajs])
            sc_rev._offsets3d = (pts[:,0], pts[:,1], pts[:,2])

        ax.view_init(elev=18, azim=35 + frame * 0.55)
        return []

    ani = animation.FuncAnimation(fig, update, frames=TOTAL,
                                  interval=1000//fps, blit=False)
    ani.save(out_path, writer="pillow", fps=fps,
             savefig_kwargs={"facecolor": "#050d1a"})
    plt.close(fig)
    print(f"  Saved animation → {out_path}")


if __name__ == "__main__":
    import os
    OUT = "/mnt/user-data/outputs"
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(67)
    psi_0 = random_qubit(np.random.default_rng(67))
    print(f"\nReference |ψ₀⟩: {psi_0.round(3)}")
    print(f"Bloch vector  : {bloch_coords(psi_0[None])[0].round(3)}\n")

    model = QuantumDiffusionRNN(
        num_tokens=256,
        embed_dim=48,
        hidden_dim=256,
        n_layers=2,
        top_k=8,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTM parameters: {n_params:,}")

    losses = train(
        model, psi_0,
        n_epochs=300,
        n_traj_per_epoch=20,
        sde_steps=200,
        seq_len=50,
        noise_strength=1.5,
        lr=3e-4,
    )

    print("\nEvaluating...")
    results = evaluate(model, psi_0, n_trials=50, n_steps=50)

    N_VIZ   = 60    
    N_FWD   = 300    
    N_REV   = 150    

    print(f"\nGenerating {N_VIZ} forward trajectories ({N_FWD} steps)...")
    rng_viz = np.random.default_rng(seed=42)
    fwd_trajs = []
    for i in range(N_VIZ):
        traj = forward_diffusion(psi_0, n_steps=N_FWD,
                                 noise_strength=2.5,
                                 rng=np.random.default_rng(1000+i))
        fwd_trajs.append(traj["psi"])

    spread = np.array([bloch_coords(t[-1][None])[0] for t in fwd_trajs])
    print(f"  Final spread: x∈[{spread[:,0].min():.2f},{spread[:,0].max():.2f}]  "
          f"z∈[{spread[:,2].min():.2f},{spread[:,2].max():.2f}]")

    print(f"Generating {N_VIZ} reverse trajectories (LSTM, {N_REV} steps)...")
    rev_trajs = []
    for i in range(N_VIZ):
        np.random.seed(2000 + i)
        rev = model.reverse_diffusion(fwd_trajs[i][-1],
                                      n_steps=N_REV, noise_scale=0.08)
        rev_trajs.append(rev["trajectory"])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N_VIZ}")

    rev_fids = np.mean([
        abs(np.vdot(rev_trajs[i][-1], psi_0))**2 for i in range(N_VIZ)])
    print(f"  Visualisation ensemble mean fidelity = {rev_fids:.4f}")

    print("\nRendering static panels...")
    make_panels(psi_0, fwd_trajs, rev_trajs,
                f"{OUT}/bloch_diffusion_panels.png")

    print("Rendering animation...")
    make_animation(psi_0, fwd_trajs[:40], rev_trajs[:40],
                   f"{OUT}/bloch_diffusion_animation.gif", fps=20)

    print(f"\n{'='*55}")
    print(f"  Final fidelity : {results['mean_fidelity']:.4f} ± {results['std_fidelity']:.4f}")
    print(f"  Bloch spread   : {results['bloch_spread']:.4f}")
    print(f"  Outputs saved to {OUT}/")
    print(f"{'='*55}")