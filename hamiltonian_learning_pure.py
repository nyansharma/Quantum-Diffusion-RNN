from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#Define our pauli basis
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=np.complex128)

#Generate a random initial Gaussian state
def random_qubit(rng=None):
    rng = np.random.default_rng() if rng is None else rng
    z = rng.normal(size=2) + 1j * rng.normal(size=2)
    return z / np.linalg.norm(z)

#Generate a trajectory based on our desired ensemble type
def generate_trajectories(count, *, n_steps, gamma, total_time, seed,
                          ensemble, fixed_state):
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(count):
        initial = fixed_state
        if ensemble == "haar":
            initial = random_qubit(rng)
        elif ensemble == "paper":
            noise = 0.2 / np.sqrt(2) * (rng.normal(size=2) + 1j * rng.normal(size=2))
            initial = normalized_qubit(np.array([1., 0.]) + noise)
        elif ensemble != "fixed":
            raise ValueError("ensemble must be fixed, haar, or paper.")
        paths.append(forward_diffusion(initial, n_steps, gamma, rng, total_time))
    return paths

#Define Bloch coordinates for tokenization and later plotting
def bloch_coords(psi):
    psi = np.asarray(psi)
    cross = np.conj(psi[..., 0]) * psi[..., 1]
    return np.stack([2 * cross.real, 2 * cross.imag,
                     abs(psi[..., 0]) ** 2 - abs(psi[..., 1]) ** 2], axis=-1)

#Normalization function for qubits to ensure they are valid quantum states
def normalized_qubit(psi):
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.shape != (2,) or not np.isfinite(psi).all():
        raise ValueError("psi_0 must be a finite complex vector of shape (2,).")
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("The zero vector is not a quantum state.")
    return psi / norm

#Weak measurement Kraus operator for simulating weak measurement 
def weak_kraus(axis, outcome, gamma, dt):
    if axis not in (0, 1, 2) or outcome not in (-1, 1):
        raise ValueError("axis must be 0/1/2 and outcome must be -1/+1.")
    axis, outcome = int(axis), int(outcome)
    if not np.isfinite([gamma, dt]).all() or gamma <= 0 or dt <= 0:
        raise ValueError("gamma and dt must be finite and positive.")
    kappa = np.sqrt(gamma * dt)
    log_norm = 0.5 * np.logaddexp(2 * kappa, -2 * kappa)
    plus = (np.eye(2) + PAULI[axis]) / 2
    minus = (np.eye(2) - PAULI[axis]) / 2
    return np.exp(outcome * kappa - log_norm) * plus + np.exp(-outcome * kappa - log_norm) * minus

#Function to forward diffuse a qubit based on the Kraus operator for weak measurement
def forward_diffusion(psi_0, n_steps=200, gamma=1.0, rng=None, total_time=1.0):
    if n_steps < 1 or not np.isfinite([total_time, gamma]).all() or total_time <= 0 or gamma <= 0:
        raise ValueError("Require n_steps>=1, finite total_time>0, and gamma>0.")
    rng = np.random.default_rng() if rng is None else rng
    dt = total_time / n_steps
    states = np.empty((n_steps + 1, 2), dtype=np.complex128)
    states[0] = normalized_qubit(psi_0)
    axes = rng.integers(0, 3, size=n_steps)
    outcomes = np.empty(n_steps, dtype=np.int64)
    kappa = np.sqrt(gamma * dt)
    kraus = {(axis, s): weak_kraus(axis, s, gamma, dt) for axis in range(3) for s in (-1, 1)}
    for k, axis in enumerate(axes):
        psi = states[k]
        sigma = PAULI[axis]
        mean = np.vdot(psi, sigma @ psi).real
        p_plus = np.clip((1 + np.tanh(2 * kappa) * mean) / 2, 0, 1)
        outcomes[k] = 1 if rng.random() < p_plus else -1
        nxt = kraus[axis, outcomes[k]] @ psi
        states[k + 1] = nxt / np.linalg.norm(nxt)
    return dict(psi=states, bloch=bloch_coords(states),
                times=np.linspace(0, total_time, n_steps + 1), axes=axes,
                outcomes=outcomes, do=outcomes * 0.5 * np.sqrt(dt / gamma),
                gamma=gamma, protocol="randomized_binary_pauli")

#Finds the shortest rotation between two Bloch vectors, returning the rotation vector (omega) that achieves this rotation. This is useful for aligning quantum states in the Bloch sphere representation.
def shortest_rotation(source, target):
    source, target = F.normalize(source, dim=-1), F.normalize(target, dim=-1)
    cross = torch.linalg.cross(source, target, dim=-1)
    sine = cross.norm(dim=-1, keepdim=True)
    cosine = (source * target).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    omega = cross * angle / sine.clamp_min(1e-10)
    basis = F.one_hot(source.abs().argmin(-1), 3).to(source.dtype)
    axis = F.normalize(torch.linalg.cross(source, basis, dim=-1), dim=-1)
    return torch.where((sine < 1e-7) & (cosine < 0), math.pi * axis, omega)

#Rotates a Bloch vector by a given rotation vector (omega) using the Rodrigues' rotation formula. This function is essential for simulating the evolution of quantum states under Hamiltonian dynamics in the Bloch sphere representation.
def rotate_bloch(bloch, omega):
    theta = omega.norm(dim=-1, keepdim=True)
    first = torch.linalg.cross(omega, bloch, dim=-1)
    second = torch.linalg.cross(omega, first, dim=-1)
    out = bloch + torch.sinc(theta / math.pi) * first
    out = out + 0.5 * torch.sinc(theta / (2 * math.pi)).square() * second
    return F.normalize(out, dim=-1)

#Defined infidelity function to measure the distance between predicted and true states in the Bloch sphere
def infidelity(pred, true):
    return (pred - true).square().sum(-1) / 4

#Defined loss function in the paper for training the model to match the Hamiltonian dynamics
def paper_score_loss(omega, previous, current, axis, gamma_dt):
    m = (previous * axis).sum(-1, keepdim=True)
    drift_correction = 2 * gamma_dt.unsqueeze(-1) * (previous - m * axis)
    residual = 2 * rotate_bloch(previous, -omega) - previous - current + drift_correction
    return residual.square().sum(-1).mean() / 4

#Class to tokenize the Bloch sphere coordinates into soft differentiable tokens
class BlochTokenizer(nn.Module):
    #initialize the tokenization process by creating centroids on the Bloch sphere and embedding them into a lower-dimensional space
    def __init__(self, num_tokens=128, embed_dim=16, temperature=20.0):
        super().__init__()
        i = torch.arange(num_tokens, dtype=torch.float32)
        z = 1 - 2 * (i + 0.5) / num_tokens
        phi = 2 * math.pi * i / ((1 + math.sqrt(5)) / 2)
        radius = torch.sqrt(1 - z.square())
        self.register_buffer("centroids", torch.stack(
            [radius * torch.cos(phi), radius * torch.sin(phi), z], dim=-1))
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.temperature = temperature
        nn.init.normal_(self.embedding.weight, std=0.05)
    #forward method computes the soft assignment of the input Bloch coordinates to the centroids and returns the corresponding embedded representation
    def forward(self, bloch):
        weights = (self.temperature * (bloch @ self.centroids.T)).softmax(-1)
        return weights @ self.embedding.weight

#Class to define the recurrent neural network that learns the quantum diffusion process, taking into account the Bloch sphere representation, temporal features, and measurement records
class QuantumDiffusionRNN(nn.Module):
    #Initialize the RNN with specified parameters, including the number of steps, mode of operation, embedding dimensions, hidden dimensions, and other configurations. The RNN is designed to process sequences of Bloch sphere coordinates and learn the underlying Hamiltonian dynamics.
    def __init__(self, n_steps=200, mode="record", num_tokens=128,
                 embed_dim=16, hidden_dim=64, n_layers=1, record_gain_offset=0.0):
        super().__init__()
        if mode not in {"single", "record", "paper"}:
            raise ValueError("mode must be 'single', 'record', or 'paper'.")
        self.config = dict(n_steps=n_steps, mode=mode, num_tokens=num_tokens,
                           embed_dim=embed_dim, hidden_dim=hidden_dim,
                           n_layers=n_layers, record_gain_offset=record_gain_offset)
        self.mode = mode
        self.tokenizer = BlochTokenizer(num_tokens, embed_dim)
        n_freq = min(n_steps // 2, 64) if mode == "single" else 4
        self.register_buffer("frequencies", torch.arange(1, n_freq + 1).float())
        input_dim = 3 + embed_dim + 2 * n_freq + 3 + 4
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 1 if mode == "record" else 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
    #Prepares the inputs for the RNN by concatenating the Bloch coordinates, tokenized embeddings, temporal features, and measurement records. The inputs are structured to provide the RNN with all necessary information for learning the quantum diffusion process.
    def _inputs(self, b, time, dt, axis, signal):
        if self.mode != "record":
            axis, signal = torch.zeros_like(axis), torch.zeros_like(signal)
        phase = 2 * math.pi * time.unsqueeze(-1) * self.frequencies
        temporal = torch.cat([time.unsqueeze(-1), dt.unsqueeze(-1),
                              torch.sqrt(dt).unsqueeze(-1),
                              phase.sin(), phase.cos()], dim=-1)
        record = torch.cat([axis, signal.unsqueeze(-1)], dim=-1)
        return torch.cat([b, self.tokenizer(b), temporal, record], dim=-1)
    #Rotates the Bloch vector based on the output of the RNN and the measurement records. The rotation is computed differently depending on the mode of operation, allowing for either a direct rotation or a learned rotation based on the RNN's output.
    def _rotation(self, out, b, axis, signal):
        if self.mode != "record":
            axis, signal = torch.zeros_like(axis), torch.zeros_like(signal)
        m = (b * axis).sum(-1)
        features = torch.stack([m, signal, m * signal, signal.square()], dim=-1)
        raw = self.head(torch.cat([out, features], dim=-1))
        if self.mode == "record":
            return (2 * signal).unsqueeze(-1) * (self.config['record_gain_offset'] + raw) * torch.linalg.cross(axis, b, dim=-1)
        omega = 0.5 * raw
        return omega - (omega * b).sum(-1, keepdim=True) * b
    #Constructs the forward sequence of Bloch vectors by passing the prepared inputs through the RNN and applying the computed rotations. The method returns the updated Bloch vectors, the rotation vectors, and the hidden state of the RNN for further processing.
    def forward_sequence(self, b, time, dt, axis, signal, hidden=None):
        out, hidden = self.rnn(self._inputs(b, time, dt, axis, signal), hidden)
        omega = self._rotation(out, b, axis, signal)
        return rotate_bloch(b, omega), omega, hidden
    #Rolls out the learned dynamics over a sequence of time steps, starting from a terminal Bloch vector. The method iteratively applies the forward sequence to generate the trajectory of Bloch vectors and the corresponding rotation vectors, returning the full sequence of states and rotations.
    def rollout(self, b_terminal, time, dt, axis, signal):
        states, rotations, hidden = [b_terminal], [], None
        b = b_terminal
        for j in range(time.shape[1]):
            nxt, omega, hidden = self.forward_sequence(
                b[:, None], time[:, j:j+1], dt[:, j:j+1],
                axis[:, j:j+1], signal[:, j:j+1], hidden)
            b = nxt[:, 0]
            states.append(b)
            rotations.append(omega[:, 0])
        return torch.stack(states, dim=1), torch.stack(rotations, dim=1)
    #Constructs the reverse diffusion process by starting from a terminal quantum state and rolling back through the learned dynamics. The method takes in the terminal state, a sequence of time steps, and optionally a measurement record, returning the reconstructed trajectory of Bloch vectors, the corresponding Hamiltonians, and other relevant information for analysis.
    @torch.no_grad()
    def reverse_diffusion(self, psi_T, times, record=None):
        self.eval()
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing.")
        n = len(times) - 1
        if self.mode == "single" and n != self.config["n_steps"]:
            raise ValueError("Single-path replay requires the trained time grid length.")
        device = next(self.parameters()).device
        duration = times[-1] - times[0]
        tr = (times[1:][::-1].copy() - times[0]) / duration
        dr = np.diff(times)[::-1].copy()
        axis = np.zeros((n, 3), np.float32)
        signal = np.zeros(n, np.float32)
        if self.mode == "record":
            if record is None:
                raise ValueError("record mode requires the observed axes, outcomes, and gamma.")
            axes = np.asarray(record["axes"])
            outcomes = np.asarray(record["outcomes"])
            gamma = float(record["gamma"])
            if (axes.shape != (n,) or outcomes.shape != (n,) or
                not np.isin(axes, [0, 1, 2]).all() or not np.isin(outcomes, [-1, 1]).all()
                or not np.isfinite(gamma) or gamma <= 0):
                raise ValueError("Invalid measurement record or rate.")
            axis = np.eye(3, dtype=np.float32)[axes.astype(int)[::-1]]
            signal = (np.sqrt(gamma * dr) * outcomes[::-1]).astype(np.float32)
        tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)[None]
        states, rotations = self.rollout(tensor(bloch_coords(normalized_qubit(psi_T))),
                                        tensor(tr), tensor(dr / duration),
                                        tensor(axis), tensor(signal))
        omega = rotations[0].cpu().numpy()
        hamiltonians = np.einsum("nk,kij->nij", omega / (2 * dr[:, None]), PAULI)
        psi = normalized_qubit(psi_T)
        kets = [psi.copy()]
        for w in omega:
            angle = np.linalg.norm(w)
            U = np.cos(angle / 2) * np.eye(2) - 0.5j * np.sinc(angle / (2 * np.pi)) * np.einsum("k,kij->ij", w, PAULI)
            psi = U @ psi
            psi /= np.linalg.norm(psi)
            kets.append(psi.copy())
        return dict(trajectory=np.asarray(kets), bloch=bloch_coords(np.asarray(kets)),
                    hamiltonians=hamiltonians, rotations=omega,
                    times=times[::-1].copy(),
                    torch_bloch=states[0].cpu().numpy())

#Converts a list of trajectory dictionaries into a structured format suitable for training and evaluation, including tensors for states, time steps, measurement axes, signals, and Hamiltonian rotations. The function ensures that the trajectories are compatible with the expected input format for the model.
def pack_trajectories(trajectories, mode, device="cpu"):
    arrays = {key: [] for key in ("states", "time", "dt", "axis", "signal", "gamma_dt")}
    for traj in trajectories:
        time = np.asarray(traj["times"])
        if traj.get("protocol") != "randomized_binary_pauli":
            raise ValueError("Use randomized binary Pauli weak-measurement trajectories.")
        duration = time[-1] - time[0]
        arrays["states"].append(traj["bloch"][::-1].copy())
        arrays["time"].append((time[:0:-1] - time[0]) / duration)
        arrays["dt"].append(np.diff(time)[::-1].copy() / duration)
        gamma_dt = traj["gamma"] * np.diff(time)[::-1].copy()
        arrays["gamma_dt"].append(gamma_dt)
        arrays["axis"].append(np.eye(3)[traj["axes"][::-1]])
        arrays["signal"].append(np.sqrt(gamma_dt) * traj["outcomes"][::-1].copy())
    data = {key: torch.tensor(np.stack(value), dtype=torch.float32, device=device)
            for key, value in arrays.items()}
    data["omega"] = shortest_rotation(data["states"][:, :-1], data["states"][:, 1:])
    return data

#rolls out the model's predictions over a sequence of time steps, starting from the initial states and using the provided time steps, measurement axes, and signals. The function returns the predicted states and the corresponding rotation vectors (omega) for further evaluation.
def model_rollout(model, data):
    return model.rollout(data["states"][:, 0], data["time"], data["dt"],
                         data["axis"], data["signal"])

#Evaluates the model's performance on a given dataset by rolling out the predictions and computing various fidelity metrics, including mean path fidelity, minimum point fidelity, and mean worst path fidelity. The function also computes the paper score loss if the model is in "paper" mode, providing a comprehensive assessment of the model's ability to learn the quantum diffusion process.
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    predicted, _ = model_rollout(model, data)
    error = infidelity(predicted[:, 1:].double(), data["states"][:, 1:].double())
    fid = 1 - error
    result = dict(mean_path_fidelity=float(fid.mean()),
                min_point_fidelity=float(fid.min()),
                mean_worst_path_fidelity=float(fid.min(dim=1).values.mean()),
                initial_state_fidelity=float(fid[:, -1].mean()),
                mean_path_infidelity=float(error.mean()))
    if model.mode == "paper":
        from scipy.optimize import linear_sum_assignment
        _, omega, _ = model.forward_sequence(data["states"][:, :-1], data["time"],
                                              data["dt"], data["axis"], data["signal"])
        result["paper_score_loss"] = float(paper_score_loss(
            omega, data["states"][:, 1:], data["states"][:, :-1], data["axis"], data["gamma_dt"]))
        target = data["states"][:, -1].double().cpu().numpy()
        for name, sample in [("initial_ensemble_w1", predicted[:, -1]),
                             ("terminal_to_initial_w1", data["states"][:, 0])]:
            points = sample.double().cpu().numpy()
            cost = np.linalg.norm(points[:, None] - target[None, :], axis=-1) / 2
            rows, cols = linear_sum_assignment(cost)
            result[name] = float(cost[rows, cols].mean())
    return result

#Trains the QuantumDiffusionRNN model using the provided training and validation datasets. The function performs optimization over a specified number of epochs, using mini-batches of data and applying gradient clipping to stabilize training. It also includes an optional rollout evaluation to assess the model's performance on longer sequences, and it saves the best model state based on validation metrics.
def train(model, training, validation, n_epochs=300, batch_size=32,
          lr=2e-3, seed=67, rollout_every=5, print_every=25):
    if n_epochs < 1 or batch_size < 1 or rollout_every < 1:
        raise ValueError("epochs, batch size, and rollout_every must be positive.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=lr * 0.05)
    rng = np.random.default_rng(seed)
    history, best = [], math.inf
    best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
    count = training["states"].shape[0]
    scale = training["omega"].square().sum(-1).mean().clamp_min(1e-6)
    warmup = max(1, n_epochs // 5)
    for epoch in range(1, n_epochs + 1):
        model.train()
        index = torch.as_tensor(rng.choice(count, min(batch_size, count), replace=False),
                                device=training["states"].device)
        data = {key: value[index] for key, value in training.items()}
        optimizer.zero_grad(set_to_none=True)
        nxt, omega, _ = model.forward_sequence(data["states"][:, :-1], data["time"],
                                               data["dt"], data["axis"], data["signal"])
        step_loss = infidelity(nxt, data["states"][:, 1:]).mean()
        h_loss = (omega - data["omega"]).square().sum(-1).mean() / scale
        if model.mode == "paper":
            loss = paper_score_loss(omega, data["states"][:, 1:], data["states"][:, :-1],
                                    data["axis"], data["gamma_dt"]) / data["gamma_dt"].mean()
        elif model.mode == "record":
            loss = step_loss / data["gamma_dt"].mean().clamp_min(1e-8)
        else:
            loss = step_loss + 0.25 * h_loss
        rollout_loss = torch.zeros((), device=loss.device)
        do_rollout = model.mode != "paper" and epoch >= warmup and (
            epoch % rollout_every == 0 or (model.mode == "single" and epoch > 0.85 * n_epochs))
        if do_rollout:
            sub = {key: value[:8] for key, value in data.items()}
            predicted, _ = model_rollout(model, sub)
            errors = infidelity(predicted[:, 1:], sub["states"][:, 1:])
            rollout_loss = errors.mean() + 0.1 * errors[:, -1].mean()
            loss = loss + 2 * rollout_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), step_loss=float(step_loss.detach()),
                   h_loss=float(h_loss.detach()), rollout_loss=float(rollout_loss.detach()))
        if epoch == 1 or epoch % print_every == 0 or epoch == n_epochs:
            metrics = evaluate(model, validation)
            row.update(metrics)
            criterion = metrics["paper_score_loss" if model.mode == "paper" else "mean_path_infidelity"]
            if criterion < best:
                best = criterion
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            summary = (f"score={metrics['paper_score_loss']:.6f}  W1={metrics['initial_ensemble_w1']:.6f}"
                       if model.mode == "paper" else
                       f"path F={metrics['mean_path_fidelity']:.6f}  min F={metrics['min_point_fidelity']:.6f}")
            print(f"Epoch {epoch:4d}/{n_epochs}  loss={row['loss']:.6f}  {summary}", flush=True)
        history.append(row)
    model.load_state_dict(best_state)
    return history, best_epoch

#Computes the inverse measurement baseline for a given trajectory by rolling back through the weak measurement process. The function reconstructs the quantum state at each time step based on the recorded measurement outcomes and axes, providing a reference for evaluating the model's performance in learning the quantum diffusion dynamics.
def inverse_measurement_baseline(traj):
    psi = traj["psi"][-1].copy()
    states = [psi.copy()]
    for axis, outcome, dt in zip(traj["axes"][::-1], traj["outcomes"][::-1], np.diff(traj["times"])[::-1]):
        a = -outcome * np.sqrt(traj["gamma"] * dt)
        plus, minus = (psi + PAULI[axis] @ psi) / 2, (psi - PAULI[axis] @ psi) / 2
        psi = np.exp(a - abs(a)) * plus + np.exp(-a - abs(a)) * minus
        psi /= np.linalg.norm(psi)
        states.append(psi.copy())
    return np.asarray(states)

#Constructs and saves a plot comparing the true trajectory of Bloch vectors with the learned reverse trajectory from the model. The plot includes a 3D representation of the Bloch sphere, fidelity metrics over time, and a comparison of the Bloch components. It also visualizes the training history to show how the model's performance evolved over epochs.
def save_plot(traj, reverse, history, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = traj["bloch"]
    predicted = reverse["bloch"][::-1]
    time = traj["times"]
    errors = np.sum((target - predicted)**2, axis=-1) / 4
    fidelity = 1 - errors
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    u, v = np.mgrid[0:2*np.pi:32j, 0:np.pi:16j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      color="#b8c5d4", alpha=0.2, linewidth=0.4)
    ax.plot(*target.T, color="#176baf", lw=2, label="Forward path")
    ax.plot(*predicted.T, color="#e47719", lw=1.5, ls="--", label="Learned reverse")
    ax.scatter(*target[0], color="#9f2241", s=40, label="Initial state")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Paired Bloch trajectories")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(2, 2, 2)
    ax.semilogy(time[:-1], np.maximum(errors[:-1], 1e-16), color="#176baf")
    ax.set(xlabel="Forward time", ylabel="Paired-state infidelity (1 - F)",
           title=f"Mean 1-F: {errors[:-1].mean():.2e} | Worst: {errors.max():.2e}")
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 2, 3)
    for j, color in enumerate(["#176baf", "#e47719", "#239365"]):
        ax.plot(time, target[:, j], color=color, label="xyz"[j])
        ax.plot(time, predicted[:, j], color=color, ls="--", alpha=0.8)
    ax.set(xlabel="Forward time", ylabel="Bloch component", title="Solid: target | Dashed: reconstruction")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 2, 4)
    rows = [r for r in history if "mean_path_infidelity" in r]
    ax.semilogy([r["epoch"] for r in rows],
                [max(1e-16, r["mean_path_infidelity"]) for r in rows], color="#176baf")
    ax.set(xlabel="Training epoch", ylabel="Free-rollout path infidelity", title="Checkpoint selection")
    ax.grid(alpha=0.2)
    fig.suptitle(label, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)

#Constructs and saves a plot specifically for the "paper" mode of the model, comparing the predicted reverse trajectory with the true forward trajectory. The plot includes a 3D representation of the Bloch sphere, empirical Wasserstein-1 distances over time, ensemble-mean Bloch components, and training history. It provides a comprehensive visualization of the model's performance in learning the quantum diffusion dynamics as described in the referenced paper.
@torch.no_grad()
def save_paper_plot(model, data, history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    model.eval()
    predicted, _ = model_rollout(model, data)
    p = predicted.cpu().numpy()
    target = data["states"].cpu().numpy()
    w1 = []
    for j in range(target.shape[1]):
        cost = np.linalg.norm(p[:, j, None] - target[None, :, j], axis=-1) / 2
        rows, cols = linear_sum_assignment(cost)
        w1.append(cost[rows, cols].mean())
    time = np.linspace(1, 0, target.shape[1])
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(*target[:, -1].T, label="Target initial ensemble", s=22, alpha=0.6)
    ax.scatter(*p[:, -1].T, label="Reconstructed ensemble", s=22, alpha=0.6)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(time, w1)
    ax.set(xlabel="Forward time / T", ylabel="Empirical W1", title="Predicted vs forward ensembles at each time")
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 2, 3)
    for j, color in enumerate(["#176baf", "#e47719", "#239365"]):
        ax.plot(time, target[:, :, j].mean(0), color=color, label="xyz"[j])
        ax.plot(time, p[:, :, j].mean(0), color=color, ls="--")
    ax.set(xlabel="Forward time / T", ylabel="Ensemble-mean Bloch component",
           title="Solid: forward | Dashed: learned reverse")
    ax.invert_xaxis()
    ax.legend(ncol=3)
    ax = fig.add_subplot(2, 2, 4)
    rows = [r for r in history if "paper_score_loss" in r]
    ax.plot([r["epoch"] for r in rows], [r["paper_score_loss"] for r in rows])
    ax.set(xlabel="Training epoch", ylabel="Validation Eq. (A26) loss", title="Checkpoint selection")
    ax.grid(alpha=0.2)
    fig.suptitle("Paper score-matching mode: empirical ensemble diagnostics")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), reverse_bloch=p,
                        target_reverse_bloch=target, empirical_w1=np.asarray(w1))

#Constructs arrays for forward and reverse trajectories, ensuring they are compatible for visualization and fidelity calculations. The function checks the dimensions and normalization of the input trajectories, and it generates time grids for both forward and reverse paths. It raises errors if the inputs do not meet the expected criteria, ensuring that the subsequent analysis can be performed correctly.
def _visual_arrays(fwd_trajs, rev_trajs, times=None, reverse_times=None):
    arrays = []
    for trajectories in (fwd_trajs, rev_trajs):
        arr = np.asarray(trajectories)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 2 or arr.shape[-1] not in (2, 3):
            raise ValueError("Expected nonempty (trajectories, times, 2 or 3) arrays.")
        arr = bloch_coords(arr) if arr.shape[-1] == 2 else np.asarray(arr, dtype=float)
        if not np.isfinite(arr).all() or not np.allclose(np.linalg.norm(arr, axis=-1), 1, atol=2e-5):
            raise ValueError("Visualizations require normalized pure states.")
        arrays.append(arr)
    fwd, rev = arrays
    if fwd.shape[0] != rev.shape[0] or not np.allclose(fwd[:, -1], rev[:, 0], atol=2e-5):
        raise ValueError("Each reverse path must start at its corresponding forward endpoint.")
    times = np.linspace(0, 1, fwd.shape[1]) if times is None else np.asarray(times, float)
    reverse_times = (np.linspace(times[-1], times[0], rev.shape[1])
                     if reverse_times is None else np.asarray(reverse_times, float))
    if (times.shape != (fwd.shape[1],) or reverse_times.shape != (rev.shape[1],)
        or not np.isfinite(times).all() or not np.isfinite(reverse_times).all()
        or np.any(np.diff(times) <= 0) or np.any(np.diff(reverse_times) >= 0)
        or not np.allclose(times[[0, -1]], reverse_times[[-1, 0]])):
        raise ValueError("Provide increasing forward times and decreasing reverse times with matching endpoints.")
    return fwd, rev, times, reverse_times

#Computes the fidelity between forward and reverse trajectories, providing metrics such as per-trajectory fidelity, mean fidelity, minimum and maximum fidelity, and mean infidelity. The function ensures that the forward and reverse trajectories are compatible in terms of time grids and normalization, allowing for accurate assessment of the model's ability to reconstruct quantum states over time.
def trajectory_fidelity(fwd_trajs, rev_trajs, times=None, reverse_times=None):
    fwd, rev, times, reverse_times = _visual_arrays(fwd_trajs, rev_trajs, times, reverse_times)
    if times.shape != reverse_times.shape or not np.allclose(times, reverse_times[::-1], rtol=0, atol=1e-10):
        raise ValueError("Fidelity requires the same forward and reverse time grid.")
    fwd = fwd.astype(np.float64)
    rev = rev[:, ::-1].astype(np.float64)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
    rev /= np.linalg.norm(rev, axis=-1, keepdims=True)
    error = np.clip(np.sum((fwd-rev)**2, axis=-1) / 4, 0, 1)
    fidelity = 1-error
    return dict(times=times.copy(), per_trajectory=fidelity,
                infidelity_per_trajectory=error, mean=fidelity.mean(axis=0),
                minimum=fidelity.min(axis=0), maximum=fidelity.max(axis=0),
                mean_infidelity=error.mean(axis=0))

#Constructs and saves a report of the fidelity metrics between forward and reverse trajectories, including CSV and NPZ files for detailed analysis. The function generates plots of the mean fidelity, minimum and maximum fidelity, and mean infidelity over time, providing a comprehensive overview of the model's performance in reconstructing quantum states from the learned dynamics.
def save_fidelity_report(report, out):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'trajectory_fidelity.npz', **report)
    t, values = report['times'], report['per_trajectory']
    with (out/'trajectory_fidelity.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'mean_fidelity', 'minimum_fidelity', 'maximum_fidelity',
                         'mean_infidelity'] + [f'F_trajectory_{i:03d}' for i in range(len(values))])
        for j, timestamp in enumerate(t):
            writer.writerow([timestamp, report['mean'][j], report['minimum'][j],
                             report['maximum'][j], report['mean_infidelity'][j], *values[:, j]])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].fill_between(t, report['minimum'], report['maximum'], alpha=.2, label='Trajectory min–max')
    axes[0].plot(t, report['mean'], label='Mean paired fidelity')
    axes[0].set(ylabel='Fidelity', ylim=(0, 1.02), title=f'Paired trajectory reconstruction ({len(values)} paths)')
    axes[0].legend(loc='lower right')
    e = report['mean_infidelity']
    if np.any(e[:-1] > 0):
        axes[1].semilogy(t[:-1], np.ma.masked_less_equal(e[:-1], 0))
    else:
        axes[1].plot(t[:-1], e[:-1])
    axes[1].set(xlabel='Physical time t (s); reverse evolves from right to left',
                ylabel='Mean infidelity (1 − F)')
    axes[1].set_title('Infidelity excludes the supplied endpoint t = T', fontsize=10)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out/'trajectory_fidelity.png', dpi=150)
    plt.close(fig)
    print(f'Paired fidelity across {len(values)} trajectories (same physical time):', flush=True)
    print('  time (s)      mean F       minimum F      mean (1-F)', flush=True)
    for fraction in (0, .25, .5, .75, 1):
        j = int(np.abs(t-(t[0]+fraction*(t[-1]-t[0]))).argmin())
        print(f"  {t[j]:8.4f}  {report['mean'][j]:.9f}  {report['minimum'][j]:.9f}  {e[j]:.3e}", flush=True)
    print('  t = T is supplied; per-step and per-trajectory values saved in trajectory_fidelity.csv/.npz.', flush=True)

#Cretes a 3D visualization of the Bloch sphere, including the sphere surface, coordinate circles, and poles. The function sets up the axes and styling for the plot, allowing for clear representation of quantum states on the Bloch sphere.
def _sphere(ax, alpha=0.10):
    u, v = np.linspace(0, 2*np.pi, 48), np.linspace(0, np.pi, 32)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)), alpha=alpha,
                    color="#1a2a3a", linewidth=0, antialiased=True, shade=False)

#Creates the coordinate circles on the Bloch sphere, representing the x-y, x-z, and y-z planes. The function plots these circles with specified line width and color, enhancing the visual representation of the Bloch sphere's geometry.
def _circles(ax):
    t = np.linspace(0, 2*np.pi, 120)
    for x, y, z in [(np.cos(t), np.sin(t), np.zeros_like(t)),
                    (np.cos(t), np.zeros_like(t), np.sin(t)),
                    (np.zeros_like(t), np.cos(t), np.sin(t))]:
        ax.plot(x, y, z, lw=0.4, color="#40556f", alpha=0.55)

#Creates the poles of the Bloch sphere, labeling the |0⟩ and |1⟩ states at the north and south poles, respectively. The function adds text labels and scatter points to indicate these quantum states on the sphere.
def _poles(ax):
    for z, label in [(1.22, "|0⟩"), (-1.22, "|1⟩")]:
        ax.text(0, 0, z, label, color="#e8e8f0", fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=20)
    ax.scatter([0, 0], [0, 0], [1, -1], c="#e8e8f0", s=18, depthshade=False)

#Creates a consistent styling for the 3D Bloch sphere plots, including axis limits, aspect ratio, tick marks, grid lines, and background color. The function also sets the view angle and adds a title if provided, ensuring that all visualizations have a uniform appearance.
def _style(ax, title="", tc="#c0d8f8"):
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), zlim=(-1.3, 1.3))
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")
    ax.set_facecolor("#050d1a")
    if title:
        ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=3)
    ax.view_init(elev=18, azim=35)

#Establishes a target point on the Bloch sphere corresponding to the initial quantum state, marking it with a distinct color and size. The function converts the initial state to Bloch coordinates and adds a scatter point to the 3D plot, providing a visual reference for the starting state of the trajectories.
def _target_dot(ax, psi_0):
    if psi_0 is not None:
        b = bloch_coords(normalized_qubit(psi_0))
        ax.scatter([b[0]], [b[1]], [b[2]], c="#ff4466", s=125, zorder=30,
                   edgecolors="white", linewidths=1.3, depthshade=False)

#Constructs a series of panels visualizing the forward and reverse trajectories on the Bloch sphere at specified snapshot fractions of the total time. The function creates a grid of subplots, displaying the state evolution and fidelity metrics, and saves the resulting figure to the specified output path. It allows for customization of labels, colors, and snapshot fractions to provide a clear comparison of the learned dynamics against the initial state.
def make_panels(psi_0, fwd_trajs, rev_trajs, out_path, *, times=None,
                reverse_times=None, reference_label="Initial state",
                mode_label="GRU Hamiltonian reversal",
                snapshot_fractions=(0, 0.25, 0.50, 0.75, 1)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    fwd, rev, times, reverse_times = _visual_arrays(fwd_trajs, rev_trajs, times, reverse_times)
    fidelity = trajectory_fidelity(fwd, rev, times, reverse_times)
    fractions = np.asarray(snapshot_fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
        or np.any(fractions < 0) or np.any(fractions > 1)
        or np.any(np.diff(fractions) <= 0)):
        raise ValueError("Use strictly increasing finite snapshot fractions between zero and one.")
    cmap_fwd = LinearSegmentedColormap.from_list("fwd", ["#2060c0", "#40a0ff", "#80d4ff"])
    cmap_rev = LinearSegmentedColormap.from_list("rev", ["#c06010", "#f0a830", "#ffe090"])
    columns, count = len(fractions), len(fwd)
    fig = plt.figure(figsize=(3.6*columns, 8), facecolor="#060e1c")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.11, wspace=0.015, hspace=0.12)
    fig.text(0.5, 0.97, "QUANTUM STATE DIFFUSION ON THE BLOCH SPHERE", ha="center",
             va="top", color="#c8dcf4", fontsize=15, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.925, f"Randomized Pauli weak measurements  |  {mode_label}  |  {count} paired trajectories",
             ha="center", color="#9cb2cd", fontsize=10)
    duration = times[-1] - times[0]
    selected_times = [times[np.abs(times - (times[0] + f * duration)).argmin()]
                      for f in fractions]
    for row, (states, grid, cmap, color) in enumerate([
        (fwd, times, cmap_fwd, "#5090d0"),
        (rev, reverse_times, cmap_rev, "#e0ac66")]):
        for col, timestamp in enumerate(selected_times):
            step = int(np.abs(grid - timestamp).argmin())
            if not np.isclose(grid[step], timestamp, rtol=0, atol=1e-10):
                raise ValueError("Forward and reverse grids must share the selected timestamps.")
            ax = fig.add_subplot(2, columns, row*columns+col+1, projection="3d")
            _sphere(ax)
            _circles(ax)
            _poles(ax)
            pts = states[:, step]
            shade = np.clip((pts[:, 2]+1)/2, 0, 1)
            ax.scatter(*pts.T, c=cmap(shade), s=25+30*shade, alpha=0.9,
                       depthshade=True, edgecolors="none")
            _target_dot(ax, psi_0)
            _style(ax, f"t = {grid[step]:.3f} s", color)
            if row == 1:
                j = int(np.abs(times-timestamp).argmin())
                summary = (f"F = {fidelity['mean'][j]:.6f}" if count == 1 else
                           f"Mean F = {fidelity['mean'][j]:.6f} | min = {fidelity['minimum'][j]:.6f}")
                summary += f"\nMean 1-F = {fidelity['mean_infidelity'][j]:.2e}"
                if j == len(times)-1:
                    summary += '\nSupplied endpoint'
                ax.text2D(.5, .02, summary, transform=ax.transAxes, ha='center', va='top',
                          color='#e0ac66', fontsize=8)
            if col == 0:
                ax.text2D(-0.06, 0.5, "FORWARD\nleft to right" if row == 0 else "REVERSE\nright to left",
                          transform=ax.transAxes, rotation=90, ha="center", va="center",
                          color=color, fontsize=10, fontweight="bold", fontfamily="monospace")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#40a0ff",
                      markersize=8, label="Forward weak measurements"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#f0a830",
                      markersize=8, label="Learned reverse evolution")]
    if psi_0 is not None:
        handles.insert(0, Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff4466",
                                 markeredgecolor="white", markersize=9, label=reference_label))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), framealpha=0.15,
               facecolor="#0a1628", edgecolor="#243040", labelcolor="#b7cbe1", fontsize=10,
               bbox_to_anchor=(0.5, 0.025))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved timestamp panels: {out_path}", flush=True)
    return fidelity

#Creates an animated GIF visualizing the forward and reverse trajectories on the Bloch sphere over time. The function generates frames for both the forward diffusion process and the learned reverse evolution, displaying the state evolution and fidelity metrics. It allows for customization of labels, frame rate, and output resolution, providing a dynamic representation of the quantum state dynamics.
def make_animation(psi_0, fwd_trajs, rev_trajs, out_path, fps=20, *, times=None,
                   reverse_times=None, reference_label="Initial state",
                   mode_label="GRU Hamiltonian reversal", phase_frames=120, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    if fps < 1 or phase_frames < 2 or dpi < 1:
        raise ValueError("fps and dpi must be positive; phase_frames must be at least 2.")
    fwd, rev, times, reverse_times = _visual_arrays(fwd_trajs, rev_trajs, times, reverse_times)
    fwd_idx = np.unique(np.rint(np.linspace(0, fwd.shape[1]-1, min(phase_frames, fwd.shape[1]))).astype(int))
    rev_idx = np.unique(np.rint(np.linspace(0, rev.shape[1]-1, min(phase_frames, rev.shape[1]))).astype(int))
    hold = max(1, round(0.9*fps))
    frames = ([('forward', int(j)) for j in fwd_idx] + [('forward', len(times)-1)]*hold
              + [('reverse', int(j)) for j in rev_idx] + [('reverse', len(reverse_times)-1)]*hold)
    fig = plt.figure(figsize=(7, 7), facecolor="#050d1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#050d1a")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _sphere(ax)
    _circles(ax)
    _poles(ax)
    _style(ax)
    _target_dot(ax, psi_0)
    phase_text = ax.text2D(0.5, 0.96, "", transform=ax.transAxes, ha="center", va="top",
                           color="#c0d8f8", fontsize=13, fontweight="bold", fontfamily="monospace")
    time_text = ax.text2D(0.5, 0.91, "", transform=ax.transAxes, ha="center", va="top",
                          color="#9bb2ce", fontsize=10, fontfamily="monospace")
    ax.text2D(0.5, 0.055, f"{len(fwd)} paired trajectories  |  {mode_label}", transform=ax.transAxes,
              ha="center", color="#9bb2ce", fontsize=9)
    if psi_0 is not None:
        ax.text2D(0.5, 0.022, f"Red marker: {reference_label}", transform=ax.transAxes,
                  ha="center", color="#e588a0", fontsize=9)
    sc_fwd = ax.scatter([], [], [], s=26, c="#40a0ff", alpha=0, depthshade=True)
    sc_rev = ax.scatter([], [], [], s=26, c="#f0a830", alpha=0, depthshade=True)

    def update(frame):
        phase, idx = frames[frame]
        forward = phase == "forward"
        sc_fwd.set_alpha(0.9 if forward else 0)
        sc_rev.set_alpha(0 if forward else 0.9)
        points = (fwd if forward else rev)[:, idx]
        artist = sc_fwd if forward else sc_rev
        artist._offsets3d = tuple(points[:, j] for j in range(3))
        phase_text.set_text("FORWARD DIFFUSION" if forward else "REVERSE DIFFUSION")
        phase_text.set_color("#4090ff" if forward else "#f0a830")
        grid = times if forward else reverse_times
        time_text.set_text(f"t = {grid[idx]:.3f} s  |  " + ("weak measurements" if forward else "GRU Hamiltonian"))
        ax.view_init(elev=18, azim=35+frame*0.55)
        return []

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    try:
        ani.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=dpi,
                 savefig_kwargs={"facecolor": "#050d1a"})
    finally:
        plt.close(fig)
    print(f"Saved forward/reverse GIF: {out_path}", flush=True)

#Saves visualizations of the Bloch sphere trajectories, including panels and animations, for a given model and set of trajectories. The function computes the predicted reverse trajectories using the model, calculates fidelity metrics, and generates visual outputs in the specified output directory. It allows for customization of ensemble types, frame rates, and other visualization parameters.
@torch.no_grad()
def save_bloch_visualizations(model, trajectories, out, *, ensemble="fixed", count=60,
                              fps=20, phase_frames=120, dpi=100, skip_animation=False):
    if count < 1:
        raise ValueError("count must be positive.")
    model.eval()
    device = next(model.parameters()).device
    data = pack_trajectories(trajectories, model.mode, device)
    predicted, _ = model_rollout(model, data)
    fwd = np.stack([t['bloch'] for t in trajectories])
    rev = predicted.cpu().numpy()
    times = trajectories[0]['times']
    if any(not np.array_equal(t['times'], times) for t in trajectories):
        raise ValueError("Visualized trajectories must use the same time grid.")
    reference, reference_label = None, ""
    if model.mode == "single" or ensemble == "fixed":
        reference, reference_label = trajectories[0]['psi'][0], "Initial state"
    elif ensemble == "paper":
        reference, reference_label = np.array([1, 0], complex), "Initial ensemble center |0⟩"
    mode_label = {'single': 'Trained-path replay', 'record': 'Measurement-conditioned reversal',
                  'paper': 'Paper score matching: ensemble recovery'}[model.mode]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fidelity = trajectory_fidelity(fwd, rev, times, times[::-1])
    save_fidelity_report(fidelity, out)
    fwd, rev = fwd[:count], rev[:count]
    np.savez_compressed(out/'bloch_diffusion_data.npz', forward_bloch=fwd, reverse_bloch=rev,
                        forward_times=times, reverse_times=times[::-1],
                        reference_state=np.asarray([] if reference is None else reference),
                        reference_label=reference_label, mode_label=mode_label)
    kwargs = dict(times=times, reverse_times=times[::-1], reference_label=reference_label,
                  mode_label=mode_label)
    make_panels(reference, fwd, rev, out/'bloch_diffusion_panels.png', **kwargs)
    if not skip_animation:
        make_animation(reference, fwd, rev, out/'bloch_diffusion_animation.gif', fps=fps,
                       phase_frames=phase_frames, dpi=dpi, **kwargs)
    return fidelity

#Loads a saved model checkpoint from the specified path, reconstructing the model architecture and loading its state dictionary. The function checks the format version of the checkpoint to ensure compatibility and returns the loaded model along with the checkpoint metadata for further analysis or visualization.
def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format_version') not in (2, 3):
        raise ValueError("Use a version 2 or 3 weak-measurement checkpoint.")
    config = dict(checkpoint['config'])
    if checkpoint['format_version'] == 2:
        config.setdefault('record_gain_offset', 1.0)
    model = QuantumDiffusionRNN(**config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint

#Loads a saved run from the specified directory, reconstructing the model and generating visualizations of the learned dynamics. The function checks the settings of the saved run, regenerates trajectories based on the original parameters, and saves Bloch sphere visualizations, including panels and animations, to the run directory. It ensures that the regenerated data matches the saved checkpoint for consistency.
def visualize_saved_run(run_directory, *, device="cpu", **visual_options):
    run_directory = Path(run_directory)
    model, checkpoint = load_checkpoint(run_directory/'model.pt', device)
    settings = json.loads((run_directory/'metrics.json').read_text())['arguments']
    if not np.isclose(settings['total_time'], 1.0, rtol=0, atol=1e-12):
        raise ValueError("This revision uses one second per phase. Retrain this run with --total-time 1; do not relabel a longer trajectory.")
    single = model.mode == 'single'
    if single:
        print('WARNING: this saved checkpoint is single-path memorization. Use examples/dynamics for a held-out ensemble.', flush=True)
    trajectories = generate_trajectories(
        1 if single else settings['test_trajectories'],
        n_steps=settings['steps'], gamma=settings['gamma'], total_time=settings['total_time'],
        seed=settings['seed'] + (1 if single else 3), ensemble=settings['ensemble'],
        fixed_state=random_qubit(np.random.default_rng(settings['seed'])))
    if not np.allclose(trajectories[0]['psi'][-1], checkpoint['psi_terminal'].cpu().numpy(), atol=1e-10):
        raise ValueError("Regenerated data does not match the saved run; check its metadata.")
    return save_bloch_visualizations(model, trajectories, run_directory,
                                    ensemble=settings['ensemble'], **visual_options)

#Main function to parse command-line arguments, set up the environment, generate trajectories, train the model, and visualize the results. The function handles different modes of operation (single, record, paper), manages random seeds for reproducibility, and ensures that the specified parameters are valid. It also provides options for visualizing saved runs without retraining.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["single", "record", "paper"], default="record",
                        help="Default record: learn conditional dynamics on independent paths. single explicitly enables memorization.")
    parser.add_argument("--steps", type=int, default=200, help="Number of transitions; states=steps+1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train-trajectories", type=int, default=1024)
    parser.add_argument("--validation-trajectories", type=int, default=128)
    parser.add_argument("--test-trajectories", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0, help="Measurement rate in Eq. (1)")
    parser.add_argument("--total-time", type=float, choices=[1.0], default=1.0,
                        help="Simulated seconds per phase: forward 0 to 1 s, then reverse 1 to 0 s (fixed at 1).")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--ensemble", choices=["fixed", "haar", "paper"], default=None,
                        help="Initial states: fixed, Haar, or a localized cloud near |0>. Default: paper except in explicit single mode.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("weak_measurement_results"))
    parser.add_argument("--visualize-only", type=Path, metavar="RUN_DIRECTORY",
                        help="Render the saved run's GIF and timestamp panels without retraining.")
    parser.add_argument("--viz-trajectories", type=int, default=128,
                        help="Maximum number of evaluated trajectories to display (single mode uses one).")
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-frames", type=int, default=120, help="Maximum frames per forward/reverse phase")
    parser.add_argument("--gif-dpi", type=int, default=100)
    parser.add_argument("--skip-animation", action="store_true", help="Save timestamp panels without rendering the GIF")
    args = parser.parse_args()
    if args.ensemble is None:
        args.ensemble = "fixed" if args.mode == "single" else "paper"
    if min(args.viz_trajectories, args.gif_fps, args.gif_dpi) < 1 or args.gif_frames < 2:
        parser.error("Visualization counts, fps, and dpi must be positive; gif-frames must be at least 2.")
    if min(args.steps, args.epochs, args.train_trajectories, args.validation_trajectories,
           args.test_trajectories, args.batch_size, args.hidden_dim, args.threads,
           args.rollout_every) < 1 or not np.isfinite([args.gamma, args.total_time, args.lr]).all() or min(args.gamma, args.total_time, args.lr) <= 0:
        parser.error("Counts, gamma, duration, and learning rate must be finite and positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    visual_options = dict(count=args.viz_trajectories, fps=args.gif_fps,
                          phase_frames=args.gif_frames, dpi=args.gif_dpi,
                          skip_animation=args.skip_animation)
    if args.visualize_only is not None:
        visualize_saved_run(args.visualize_only, device=device, **visual_options)
        return
    psi_0 = random_qubit(np.random.default_rng(args.seed))

    def generate(count, seed, ensemble=None):
        return generate_trajectories(count, n_steps=args.steps, gamma=args.gamma,
                                     total_time=args.total_time, seed=seed,
                                     ensemble=args.ensemble if ensemble is None else ensemble,
                                     fixed_state=psi_0)

    if args.mode == "single":
        print("WARNING: explicit single mode memorizes one training path and plots one point. Use default record mode for held-out dynamics.", flush=True)
        train_paths = generate(1, args.seed + 1)
        validation_paths = test_paths = train_paths
        label = "Single trained trajectory: reconstruction (not held-out generalization)"
    else:
        train_paths = generate(args.train_trajectories, args.seed + 1)
        validation_paths = generate(args.validation_trajectories, args.seed + 2)
        test_paths = generate(args.test_trajectories, args.seed + 3)
        label = ("Paper Eq. (A26): held-out ensemble recovery" if args.mode == "paper" else
                 "Measurement-conditioned reconstruction: held-out trajectories")
    train_data, val_data, test_data = [pack_trajectories(p, args.mode, device)
                                     for p in [train_paths, validation_paths, test_paths]]
    model = QuantumDiffusionRNN(args.steps, args.mode, hidden_dim=args.hidden_dim).to(device)
    print(label, flush=True)
    print(f"Device={device}; transitions={args.steps}; parameters="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    before = evaluate(model, test_data)
    history, best_epoch = train(model, train_data, val_data, args.epochs, args.batch_size,
                                args.lr, args.seed, args.rollout_every)
    result = dict(mode=args.mode, protocol="randomized_binary_pauli", paper="arXiv:2508.08799v4",
                  objective="Appendix A Eq. (A26)" if args.mode == "paper" else "paired trajectory reconstruction",
                  state_access="simulated states; no shadow decoder", evaluation=label,
                  best_epoch=best_epoch, before_training=before, after_training=evaluate(model, test_data))
    result['split_seeds'] = (dict(train=args.seed+1, validation=args.seed+1, test=args.seed+1)
                             if args.mode == 'single' else
                             dict(train=args.seed+1, validation=args.seed+2, test=args.seed+3))
    result['controller_initialization'] = 'zero_rotation'
    result['reverse_inputs'] = ('terminal state, time grid, axes and outcomes' if args.mode == 'record'
                                else 'terminal state and time grid')
    cloud = np.stack([p['bloch'] for p in test_paths])
    result['forward_spreading'] = dict(
        initial_mean_bloch_length=float(np.linalg.norm(cloud[:, 0].mean(axis=0))),
        terminal_mean_bloch_length=float(np.linalg.norm(cloud[:, -1].mean(axis=0))),
        terminal_bloch_covariance=np.cov(cloud[:, -1].T).tolist() if len(cloud)>1 else None)
    identity = test_data["states"][:, :1].expand_as(test_data["states"])
    result["identity_mean_path_fidelity"] = float(1 - infidelity(identity[:, 1:], test_data["states"][:, 1:]).mean())
    if args.mode == "record":
        tampered = dict(test_data)
        tampered['states'] = test_data['states'].clone()
        tampered['states'][:, 1:] = -tampered['states'][:, 1:]
        with torch.no_grad():
            original_prediction, _ = model_rollout(model, test_data)
            changed_prediction, _ = model_rollout(model, tampered)
        result['target_independence_max_change'] = float((original_prediction-changed_prediction).abs().max().detach())
        shuffled = dict(test_data)
        order = torch.arange(args.steps - 1, -1, -1, device=device)
        shuffled["signal"] = test_data["signal"][:, order]
        result["wrong_record"] = evaluate(model, shuffled)
        absent = dict(test_data)
        absent['signal'] = torch.zeros_like(test_data['signal'])
        result['no_record'] = evaluate(model, absent)
        unseen = pack_trajectories(generate(args.test_trajectories, args.seed + 4, "haar"), args.mode, device)
        result["unseen_initial_states"] = evaluate(model, unseen)
    elif args.mode == "single":
        unseen = pack_trajectories(generate(8, args.seed + 4), args.mode, device)
        result["unseen_paths_diagnostic_only"] = evaluate(model, unseen)
    example = test_paths[0]
    record = {k: example[k] for k in ["axes", "outcomes", "gamma"]} if args.mode == "record" else None
    reverse = model.reverse_diffusion(example["psi"][-1], example["times"], record)
    exact = inverse_measurement_baseline(example)
    result["inverse_kraus_baseline_min_fidelity"] = float(np.min(
        abs(np.sum(exact.conj() * example["psi"][::-1], axis=-1))**2))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=3, protocol="randomized_binary_pauli", config=model.config, state_dict=model.state_dict(),
                    times=torch.tensor(example["times"]),
                    psi_terminal=torch.tensor(example["psi"][-1]),
                    axes=torch.tensor(example["axes"]), outcomes=torch.tensor(example["outcomes"]),
                    gamma=float(example["gamma"]), best_epoch=best_epoch), args.out / "model.pt")
    np.savez_compressed(args.out / "trajectory.npz", forward_psi=example["psi"],
                        forward_times=example["times"], reverse_psi=reverse["trajectory"],
                        reverse_times=reverse["times"], reverse_hamiltonians=reverse["hamiltonians"],
                        axes=example["axes"], outcomes=example["outcomes"],
                        do=example["do"], gamma=example["gamma"])
    serial_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "metrics.json").write_text(json.dumps(dict(arguments=serial_args, **result), indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if args.mode == "paper":
        save_paper_plot(model, test_data, history, args.out / "ensemble.png")
    else:
        save_plot(example, reverse, history, args.out / "trajectory.png", label)
    save_bloch_visualizations(model, test_paths, args.out, ensemble=args.ensemble, **visual_options)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved checkpoint, paired trajectory, metrics, and plot to {args.out.resolve()}")


if __name__ == "__main__":
    main()
