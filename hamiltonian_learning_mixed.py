from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)

#Verifies that rho is a density matrix by checking its shape, finiteness, Hermitian property, trace, and positive semidefiniteness. Raises ValueError if any condition is not met.
def validate_density(rho):
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (2, 2) or not np.isfinite(rho).all():
        raise ValueError("rho must be a finite 2 by 2 matrix.")
    if not np.allclose(rho, rho.conj().T, atol=1e-10, rtol=0):
        raise ValueError("rho must be Hermitian.")
    if not np.isclose(np.trace(rho), 1, atol=1e-10, rtol=0):
        raise ValueError("rho must have trace one.")
    if np.linalg.eigvalsh(rho).min() < -1e-10:
        raise ValueError("rho must be positive semidefinite.")
    return rho.copy()

#Converts a density matrix to its corresponding Bloch vector coordinates by taking the expectation values of the Pauli matrices. The function uses Einstein summation to efficiently compute the dot product between the density matrix and the Pauli matrices, returning the real part of the resulting Bloch vector.
def bloch_coords(rho):
    return np.einsum("...ij,kji->...k", np.asarray(rho), PAULI).real

#Converts a Bloch vector to its corresponding density matrix representation. The function checks that the input Bloch vectors are finite and lie within the unit ball, then constructs the density matrix using the identity matrix and the Pauli matrices, ensuring that the resulting density matrix is valid.
def density_from_bloch(b):
    b = np.asarray(b, dtype=float)
    if b.shape[-1] != 3 or not np.isfinite(b).all():
        raise ValueError("Bloch vectors must be finite with final dimension 3.")
    if np.any(np.linalg.norm(b, axis=-1) > 1 + 1e-9):
        raise ValueError("Bloch vectors must lie inside the unit ball.")
    return (IDENTITY + np.einsum("...k,kij->...ij", b, PAULI)) / 2

#Defines the weak Kraus operator for a given measurement axis, outcome, measurement strength (gamma), and time step (dt). The function computes the Kraus operator based on the specified parameters, ensuring that the axis and outcome are valid and that gamma and dt are positive and finite. The resulting operator is a linear combination of projectors onto the eigenstates of the Pauli matrix corresponding to the specified axis, scaled by exponential factors determined by the measurement strength and outcome.
def weak_kraus(axis, outcome, gamma, dt):

    if axis not in (0, 1, 2) or outcome not in (-1, 1):
        raise ValueError("Invalid measurement axis or outcome.")
    if not np.isfinite([gamma, dt]).all() or min(gamma, dt) <= 0:
        raise ValueError("gamma and dt must be positive and finite.")
    kappa = np.sqrt(gamma * dt)
    log_norm = 0.5 * np.logaddexp(2 * kappa, -2 * kappa)
    plus, minus = (IDENTITY + PAULI[axis]) / 2, (IDENTITY - PAULI[axis]) / 2
    return (np.exp(outcome * kappa - log_norm) * plus
            + np.exp(-outcome * kappa - log_norm) * minus)

#Simulates the forward diffusion of a quantum state under randomized weak Pauli measurements. The function takes an initial density matrix (rho_0), the number of time steps (n_steps), measurement strength (gamma), a random number generator (rng), and the total simulation time (total_time). It generates a sequence of density matrices representing the state evolution, along with the corresponding Bloch coordinates, measurement axes, outcomes, and times. The function returns a dictionary containing the simulation results.
def forward_diffusion(rho_0, n_steps=200, gamma=1., rng=None, total_time=1.):
    if n_steps < 1 or not np.isfinite([gamma, total_time]).all() or min(gamma, total_time) <= 0:
        raise ValueError("Require n_steps >= 1 and positive finite gamma and total_time.")
    rng = np.random.default_rng() if rng is None else rng
    dt = total_time / n_steps
    states = np.empty((n_steps + 1, 2, 2), dtype=np.complex128)
    states[0] = validate_density(rho_0)
    axes = rng.integers(0, 3, size=n_steps)
    outcomes = np.empty(n_steps, dtype=np.int64)
    operators = {(a, s): weak_kraus(a, s, gamma, dt)
                 for a in range(3) for s in (-1, 1)}
    for k, axis in enumerate(axes):
        kp = operators[axis, 1]
        unnormalized_plus = kp @ states[k] @ kp.conj().T
        p_plus = float(np.trace(unnormalized_plus).real)
        outcomes[k] = 1 if rng.random() < np.clip(p_plus, 0., 1.) else -1
        operator = operators[axis, outcomes[k]]
        unnormalized = operator @ states[k] @ operator.conj().T
        probability = float(np.trace(unnormalized).real)
        if probability <= 0:
            raise FloatingPointError("Sampled a zero-probability measurement outcome.")
        nxt = unnormalized / probability
        states[k + 1] = (nxt + nxt.conj().T) / 2
    return dict(rho=states, bloch=bloch_coords(states),
                times=np.linspace(0, total_time, n_steps + 1), axes=axes,
                outcomes=outcomes, gamma=float(gamma),
                protocol="randomized_binary_pauli_mixed")

#Generates a specified number of quantum state trajectories under randomized weak Pauli measurements. The function allows for different ensembles of initial states, including fixed states, random states within a Bloch ball, and states following a specific distribution. It takes parameters such as the number of steps, measurement strength, total time, random seed, and ensemble type. The function returns a list of trajectory dictionaries containing the simulation results for each generated trajectory.
def generate_trajectories(count, *, n_steps, gamma, total_time, seed,
                          ensemble="ball", radius=.6, radius_min=0., radius_max=.9,
                          fixed_state=None):
    if not (0 <= radius <= 1 and 0 <= radius_min <= radius_max <= 1):
        raise ValueError("All radii must be in [0,1] with min <= max.")
    rng = np.random.default_rng(seed)
    fixed = density_from_bloch([0., 0., radius]) if fixed_state is None else validate_density(fixed_state)
    paths = []
    for _ in range(count):
        if ensemble == "fixed":
            initial = fixed
        elif ensemble == "ball":
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            r = (radius_min**3 + rng.random() * (radius_max**3 - radius_min**3))**(1/3)
            initial = density_from_bloch(r * direction)
        elif ensemble == "paper":
            psi = np.array([1., 0.], complex) + .2 / np.sqrt(2) * (
                rng.normal(size=2) + 1j * rng.normal(size=2))
            psi /= np.linalg.norm(psi)
            initial = radius * np.outer(psi, psi.conj()) + (1 - radius) * IDENTITY / 2
        else:
            raise ValueError("ensemble must be fixed, ball, or paper.")
        paths.append(forward_diffusion(initial, n_steps, gamma, rng, total_time))
    return paths

# Packs a list of trajectory dictionaries into a single dictionary of tensors suitable for training or evaluation. The function extracts the relevant data from each trajectory, including Bloch coordinates, time, time steps, measurement axes, measurement signals, and scaled measurement strengths. It normalizes the time and time step values by the total duration of each trajectory and converts the data into PyTorch tensors for efficient computation on the specified device (CPU or GPU).
def pack_trajectories(paths, device="cpu"):
    arrays = {k: [] for k in ("states", "time", "dt", "axis", "signal", "gamma_dt")}
    for path in paths:
        times = path["times"]
        duration = times[-1] - times[0]
        dt = np.diff(times)[::-1].copy()
        gd = path["gamma"] * dt
        arrays["states"].append(path["bloch"][::-1].copy())
        arrays["time"].append((times[:0:-1] - times[0]) / duration)
        arrays["dt"].append(dt / duration)
        arrays["gamma_dt"].append(gd)
        arrays["axis"].append(np.eye(3)[path["axes"][::-1]])
        arrays["signal"].append(np.sqrt(gd) * path["outcomes"][::-1])
    return {k: torch.as_tensor(np.stack(v), dtype=torch.float64, device=device)
            for k, v in arrays.items()}

#Defines a loss function for comparing predicted and true Bloch vectors. The function computes the squared difference between the predicted and true vectors, sums the squared differences across the last dimension, and normalizes the result by dividing by 4. This loss function is suitable for training models that predict quantum states represented as Bloch vectors.
def state_loss(pred, true):
    return (pred - true).square().sum(-1) / 4

#Defines a fidelity function for comparing predicted and true Bloch vectors. The function computes the fidelity between the predicted and true states using the formula for fidelity in terms of Bloch vectors. It calculates the inner product of the predicted and true vectors, as well as the norms of the vectors, and returns a value between 0 and 1 representing the fidelity of the predicted state with respect to the true state.
def fidelity(pred, true):
    a = (1 - pred.square().sum(-1)).clamp_min(0)
    b = (1 - true.square().sum(-1)).clamp_min(0)
    return ((1 + (pred * true).sum(-1) + torch.sqrt(a * b)) / 2).clamp(0, 1)

#Tokenizes Bloch vectors into discrete tokens for embedding in a neural network. The BlochTokenizer class creates a set of centroids on the Bloch sphere and computes the softmax-weighted embedding of the input Bloch vectors based on their distances to these centroids. The resulting embeddings can be used as input features for models that learn to predict quantum state dynamics.
class BlochTokenizer(nn.Module):
    #Initializes the BlochTokenizer with a specified number of tokens and embedding dimension. The constructor generates centroids on the Bloch sphere using spherical coordinates and registers them as a buffer. It also creates an embedding layer that maps the discrete tokens to continuous embeddings, initializing the weights with a normal distribution.
    def __init__(self, num_tokens=128, embed_dim=16):
        super().__init__()
        i = torch.arange(num_tokens, dtype=torch.float64)
        z = 1 - 2 * (i + .5) / num_tokens
        phi = 2 * math.pi * i / ((1 + math.sqrt(5)) / 2)
        rxy = torch.sqrt(1 - z.square())
        directions = torch.stack([rxy * phi.cos(), rxy * phi.sin(), z], dim=-1)
        radii = (i.remainder(4) + 1) / 4
        centers = directions * radii[:, None]
        centers[0] = 0
        self.register_buffer("centroids", centers)
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        nn.init.normal_(self.embedding.weight, std=.05)
    #Computes the softmax-weighted embedding of the input Bloch vectors based on their distances to the centroids. The forward method calculates the squared distances between the input Bloch vectors and the centroids, applies a softmax function to obtain weights, and then computes the weighted sum of the embedding vectors corresponding to the centroids. This results in a continuous embedding representation of the input Bloch vectors.
    def forward(self, b):
        squared_distances = (b[..., None, :] - self.centroids).square().sum(-1)
        return (-10 * squared_distances).softmax(-1) @ self.embedding.weight

#Applies a filtering operation to the Bloch vectors based on the specified measurement axis and strength. The filter_bloch function computes the parallel and perpendicular components of the Bloch vectors with respect to the measurement axis, applies a hyperbolic tangent transformation to the strength, and combines the components to produce the filtered Bloch vectors. The result is normalized to ensure that the Bloch vectors remain within the unit ball, while only removing floating-point overshoots without affecting interior radii.
def filter_bloch(b, axis, strength):
    m = (b * axis).sum(-1, keepdim=True)
    a = strength.unsqueeze(-1)
    q = torch.tanh(2 * a)
    denominator = 1 + m * q
    parallel = ((m + q) / denominator) * axis
    perpendicular = (b - m * axis) / (torch.cosh(2 * a) * denominator)
    result = parallel + perpendicular
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1)

#Defines a recurrent neural network model for learning the dynamics of mixed quantum states under weak measurements. The MixedStateDiffusionRNN class combines a Bloch tokenizer, a GRU-based RNN, and a feedforward head to predict the evolution of Bloch vectors over time. The model takes as input the current Bloch vector, time, time step, measurement axis, and measurement signal, and outputs the next Bloch vector along with the gain applied during the update. The model can also perform rollouts and reverse diffusion based on recorded measurement data.
class MixedStateDiffusionRNN(nn.Module):
    #Initializes the MixedStateDiffusionRNN with specified parameters for the number of tokens, embedding dimension, hidden dimension, and number of layers. The constructor sets up the Bloch tokenizer, registers frequency buffers, creates a GRU-based RNN for processing the input sequences, and defines a feedforward head for predicting the gain. The model is configured to use double precision for numerical stability.
    def __init__(self, num_tokens=128, embed_dim=16, hidden_dim=64, n_layers=1):
        super().__init__()
        self.config = dict(num_tokens=num_tokens, embed_dim=embed_dim,
                           hidden_dim=hidden_dim, n_layers=n_layers)
        self.tokenizer = BlochTokenizer(num_tokens, embed_dim)
        self.register_buffer("frequencies", torch.arange(1, 5, dtype=torch.float64))
        self.rnn = nn.GRU(3 + embed_dim + 11 + 4, hidden_dim,
                          num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 5, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.double()
    #Defines the forward pass of the model for a sequence of inputs. The forward_sequence method takes the current Bloch vector, time, time step, measurement axis, and measurement signal as inputs, along with an optional hidden state. It computes the phase information based on the frequencies, concatenates the inputs, and passes them through the RNN. The method then calculates the gain and strength based on the RNN output and applies the filtering operation to update the Bloch vector. The updated Bloch vector, gain, and hidden state are returned.
    def forward_sequence(self, b, time, dt, axis, signal, hidden=None):
        phase = 2 * math.pi * time[..., None] * self.frequencies
        temporal = torch.cat([time[..., None], dt[..., None], dt.sqrt()[..., None],
                              phase.sin(), phase.cos()], dim=-1)
        inputs = torch.cat([b, self.tokenizer(b), temporal, axis, signal[..., None]], dim=-1)
        out, hidden = self.rnn(inputs, hidden)
        m = (b * axis).sum(-1)
        features = torch.stack([m, signal, m * signal, signal.square(),
                                b.square().sum(-1)], dim=-1)
        raw = self.head(torch.cat([out, features], dim=-1)).squeeze(-1)
        gain = 2 * torch.tanh(raw)  # zero initially; exact inverse has gain=1
        strength = -signal * gain
        return filter_bloch(b, axis, strength), gain, hidden
    #Performs a rollout of the model over a sequence of time steps, starting from a given terminal Bloch vector. The rollout method iteratively applies the forward_sequence method for each time step, updating the Bloch vector and recording the gains. The resulting sequence of Bloch vectors and gains is returned as stacked tensors.
    def rollout(self, b_terminal, time, dt, axis, signal):
        b, hidden = b_terminal, None
        states, gains = [b], []
        for j in range(time.shape[1]):
            nxt, gain, hidden = self.forward_sequence(
                b[:, None], time[:, j:j+1], dt[:, j:j+1],
                axis[:, j:j+1], signal[:, j:j+1], hidden)
            b = nxt[:, 0]
            states.append(b)
            gains.append(gain[:, 0])
        return torch.stack(states, dim=1), torch.stack(gains, dim=1)
    #Performs reverse diffusion of a quantum state based on a recorded measurement history. The reverse_diffusion method takes a terminal density matrix, an array of times, and a measurement record as inputs. It validates the inputs, extracts the relevant data from the record, and performs a rollout of the model in reverse time order. The resulting trajectory of Bloch vectors, gains, and filter operators is returned in a dictionary format.
    @torch.no_grad()
    def reverse_diffusion(self, rho_T, times, record):
        self.eval()
        rho_T = validate_density(rho_T)
        times = np.asarray(times, float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing.")
        n = len(times) - 1
        axes, outcomes = np.asarray(record["axes"]), np.asarray(record["outcomes"])
        gamma = float(record["gamma"])
        if (axes.shape != (n,) or outcomes.shape != (n,) or
            not np.isin(axes, [0, 1, 2]).all() or not np.isin(outcomes, [-1, 1]).all()
            or not np.isfinite(gamma) or gamma <= 0):
            raise ValueError("Invalid measurement record.")
        device = next(self.parameters()).device
        tensor = lambda v: torch.as_tensor(np.array(v), dtype=torch.float64, device=device)[None]
        duration = times[-1] - times[0]
        dr = np.diff(times)[::-1].copy()
        reverse_axes = axes.astype(int)[::-1]
        signal = np.sqrt(gamma * dr) * outcomes[::-1]
        states, gains = self.rollout(tensor(bloch_coords(rho_T)),
                                    tensor((times[:0:-1] - times[0]) / duration),
                                    tensor(dr / duration), tensor(np.eye(3)[reverse_axes]), tensor(signal))
        b, g = states[0].cpu().numpy(), gains[0].cpu().numpy()
        a = -signal * g
        filters = (np.cosh(a)[:, None, None] * IDENTITY
                   + np.sinh(a)[:, None, None] * PAULI[reverse_axes])
        return dict(trajectory=density_from_bloch(b), bloch=b, filters=filters,
                    gains=g, strengths=a, times=times[::-1].copy())

# Rolls out the model over a sequence of time steps, starting from the initial Bloch vector in the provided data. The model_rollout function extracts the relevant data from the input dictionary and calls the rollout method of the model, returning the resulting sequence of Bloch vectors and gains.
def model_rollout(model, data):
    return model.rollout(data["states"][:, 0], data["time"], data["dt"],
                         data["axis"], data["signal"])

# Evaluates the model's performance on a given dataset by computing various metrics, including mean path fidelity, minimum point fidelity, initial state fidelity, mean path state loss, mean path purity error, mean gain, and mean gain error. The evaluate function sets the model to evaluation mode, performs a rollout using the model_rollout function, and calculates the metrics based on the predicted and target Bloch vectors.
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    predicted, gains = model_rollout(model, data)
    target = data["states"]
    values = fidelity(predicted[:, 1:], target[:, 1:])
    purity_error = (predicted[:, 1:].square().sum(-1) - target[:, 1:].square().sum(-1)).abs() / 2
    return dict(mean_path_fidelity=float(values.mean()), min_point_fidelity=float(values.min()),
                initial_state_fidelity=float(values[:, -1].mean()),
                mean_path_state_loss=float(state_loss(predicted[:, 1:], target[:, 1:]).mean()),
                mean_path_purity_error=float(purity_error.mean()),
                mean_gain=float(gains.mean()), mean_gain_error=float((gains - 1).abs().mean()))
#Trains the model using the provided training and validation datasets. The train function performs multiple epochs of training, shuffling the training data, computing the loss, and updating the model parameters using the AdamW optimizer. It also evaluates the model on the validation set at specified intervals and keeps track of the best-performing model based on validation metrics. The function returns the training history and the epoch of the best model.
def train(model, training, validation, n_epochs=100, batch_size=32, lr=2e-3,
          seed=67, rollout_every=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=lr * .05)
    rng = np.random.default_rng(seed)
    count = len(training["states"])
    best, best_epoch, best_state = math.inf, 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(count)
        losses = []
        for start in range(0, count, batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], device=training["states"].device)
            data = {k: v[indices] for k, v in training.items()}
            optimizer.zero_grad(set_to_none=True)
            nxt, _, _ = model.forward_sequence(data["states"][:, :-1], data["time"],
                                               data["dt"], data["axis"], data["signal"])
            loss = state_loss(nxt, data["states"][:, 1:]).mean() / data["gamma_dt"].mean().clamp_min(1e-12)
            if epoch % rollout_every == 0:
                sub = {k: v[:8] for k, v in data.items()}
                predicted, _ = model_rollout(model, sub)
                errors = state_loss(predicted[:, 1:], sub["states"][:, 1:])
                loss = loss + 2 * (errors.mean() + .1 * errors[:, -1].mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = dict(epoch=epoch, loss=float(np.mean(losses)))
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            metrics = evaluate(model, validation)
            row.update(metrics)
            if metrics["mean_path_state_loss"] < best:
                best = metrics["mean_path_state_loss"]
                best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch:4d}/{n_epochs}: loss={row['loss']:.5g}, "
                  f"validation path F={metrics['mean_path_fidelity']:.7f}, "
                  f"initial F={metrics['initial_state_fidelity']:.7f}", flush=True)
        history.append(row)
    model.load_state_dict(best_state)
    return history, best_epoch

#Defines a baseline method for reconstructing the quantum state trajectory from the final density matrix and the recorded measurement history. The inverse_measurement_baseline function iteratively applies the inverse of the weak measurement operators in reverse time order, updating the density matrix at each step. The resulting sequence of density matrices is returned as an array, representing the reconstructed trajectory of the quantum state.
def inverse_measurement_baseline(path):
    rho, states = path["rho"][-1].copy(), [path["rho"][-1].copy()]
    for axis, outcome, dt in zip(path["axes"][::-1], path["outcomes"][::-1],
                                 np.diff(path["times"])[::-1]):
        a = -outcome * np.sqrt(path["gamma"] * dt)
        operator = np.cosh(a) * IDENTITY + np.sinh(a) * PAULI[axis]
        rho = operator @ rho @ operator.conj().T
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real
        states.append(rho.copy())
    return np.asarray(states)

#Ensures that the forward and reverse trajectories are valid Bloch vectors or density matrices, and that they have matching endpoints. The _visual_arrays function checks the dimensions and properties of the input trajectories, converts density matrices to Bloch coordinates if necessary, and verifies that the forward and reverse trajectories start and end at the same points. It also checks that the provided time arrays are strictly increasing for the forward trajectory and strictly decreasing for the reverse trajectory, with matching endpoints. The function returns the validated forward and reverse trajectories along with their corresponding time arrays.
def _visual_arrays(fwd_trajs, rev_trajs, times=None, reverse_times=None):
    arrays = []
    for trajectories in (fwd_trajs, rev_trajs):
        arr = np.asarray(trajectories)
        if arr.ndim >= 3 and arr.shape[-2:] == (2, 2):
            arr = bloch_coords(arr)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 2 or arr.shape[-1] != 3:
            raise ValueError("Expected (trajectories, times, 3) Bloch vectors or density matrices.")
        arr = np.asarray(arr, dtype=float)
        if not np.isfinite(arr).all() or np.any(np.linalg.norm(arr, axis=-1) > 1 + 2e-5):
            raise ValueError("Mixed-state Bloch vectors must lie inside the unit ball.")
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
        raise ValueError("Provide increasing forward and decreasing reverse times with matching endpoints.")
    return fwd, rev, times, reverse_times

#Computes the fidelity between forward and reverse trajectories at matching time points. The trajectory_fidelity function takes forward and reverse trajectories, along with optional time arrays, and validates them using the _visual_arrays function. It then computes the fidelity between the forward and reversed reverse trajectories using the fidelity function, returning a dictionary containing the time points, per-trajectory fidelities, infidelities, and summary statistics such as mean, minimum, and maximum fidelities.
def trajectory_fidelity(fwd_trajs, rev_trajs, times=None, reverse_times=None):
    fwd, rev, times, reverse_times = _visual_arrays(fwd_trajs, rev_trajs, times, reverse_times)
    if times.shape != reverse_times.shape or not np.allclose(times, reverse_times[::-1], rtol=0, atol=1e-10):
        raise ValueError("Fidelity requires matching forward and reverse time grids.")
    values = fidelity(torch.from_numpy(fwd.copy()), torch.from_numpy(rev[:, ::-1].copy())).numpy()
    error = 1 - values
    return dict(times=times.copy(), per_trajectory=values, infidelity_per_trajectory=error,
                mean=values.mean(0), minimum=values.min(0), maximum=values.max(0),
                mean_infidelity=error.mean(0))

# Saves the fidelity report to disk in both compressed NumPy format and CSV format, and generates a plot of the trajectory fidelity over time. The save_fidelity_report function takes a report dictionary and an output directory, creates the directory if it doesn't exist, saves the report data, writes a CSV file with the fidelity metrics, and generates a plot showing the mean fidelity, minimum and maximum fidelities, and mean infidelity over time. The function also prints a summary of the fidelity metrics at selected time points.
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
# Creates a 3D surface plot of the Bloch sphere with specified transparency. The _sphere function generates a mesh grid of spherical coordinates and plots the surface of the sphere using the provided axes object. The function allows for customization of the transparency (alpha) of the sphere, enhancing the visual representation of quantum states on the Bloch sphere.
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
def _target_dot(ax, rho_0):
    if rho_0 is not None:
        b = bloch_coords(validate_density(rho_0))
        ax.scatter([b[0]], [b[1]], [b[2]], c="#ff4466", s=125, zorder=30,
                   edgecolors="white", linewidths=1.3, depthshade=False)

#Constructs a series of panels visualizing the forward and reverse trajectories on the Bloch sphere at specified snapshot fractions of the total time. The function creates a grid of subplots, displaying the state evolution and fidelity metrics, and saves the resulting figure to the specified output path. It allows for customization of labels, colors, and snapshot fractions to provide a clear comparison of the learned dynamics against the initial state.
def make_panels(rho_0, fwd_trajs, rev_trajs, out_path, *, times=None,
                reverse_times=None, reference_label="Initial state",
                mode_label="GRU nonunitary reversal",
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
    fig.text(0.5, 0.97, "MIXED-STATE DIFFUSION IN THE BLOCH BALL", ha="center",
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
            _target_dot(ax, rho_0)
            _style(ax, f"t = {grid[step]:.3f} s", color)
            if row == 1:
                j = int(np.abs(times-timestamp).argmin())
                summary = (f"F = {fidelity['mean'][j]:.6f}" if count == 1 else
                           f"Mean F = {fidelity['mean'][j]:.6f} | min = {fidelity['minimum'][j]:.6f}")
                summary += f"\nMean 1-F = {fidelity['mean_infidelity'][j]:.2e}"
                summary += f"\nMean purity = {((1 + (pts**2).sum(-1)) / 2).mean():.4f}"
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
    if rho_0 is not None:
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
def make_animation(rho_0, fwd_trajs, rev_trajs, out_path, fps=20, *, times=None,
                   reverse_times=None, reference_label="Initial state",
                   mode_label="GRU nonunitary reversal", phase_frames=120, dpi=100):
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
    _target_dot(ax, rho_0)
    phase_text = ax.text2D(0.5, 0.96, "", transform=ax.transAxes, ha="center", va="top",
                           color="#c0d8f8", fontsize=13, fontweight="bold", fontfamily="monospace")
    time_text = ax.text2D(0.5, 0.91, "", transform=ax.transAxes, ha="center", va="top",
                          color="#9bb2ce", fontsize=10, fontfamily="monospace")
    ax.text2D(0.5, 0.055, f"{len(fwd)} paired trajectories  |  {mode_label}", transform=ax.transAxes,
              ha="center", color="#9bb2ce", fontsize=9)
    if rho_0 is not None:
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
        purity = ((1 + (points**2).sum(-1)) / 2).mean()
        time_text.set_text(f"t = {grid[idx]:.3f} s  |  mean purity = {purity:.4f}")
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

def save_bloch_visualizations(forward, reverse, out, *, times, reverse_times=None,
                              reference_state=None, reference_label="Initial mixed state",
                              count=128, fps=20, phase_frames=120, dpi=100,
                              skip_animation=False,
                              snapshot_fractions=(0, .25, .5, .75, 1)):
    """Original pure-process styling, adapted to density matrices and mixed fidelity."""
    if min(count, fps, dpi) < 1 or phase_frames < 2:
        raise ValueError("Counts, fps, dpi must be positive; phase_frames must be at least 2.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    forward, reverse, times, reverse_times = _visual_arrays(forward, reverse, times, reverse_times)
    report = trajectory_fidelity(forward, reverse, times, reverse_times)
    save_fidelity_report(report, out)
    kwargs = dict(times=times, reverse_times=reverse_times,
                  reference_label=reference_label, mode_label="Measurement-conditioned reversal")
    make_panels(reference_state, forward[:count], reverse[:count],
                out / "bloch_diffusion_panels.png", snapshot_fractions=snapshot_fractions, **kwargs)
    if not skip_animation:
        make_animation(reference_state, forward[:count], reverse[:count],
                       out / "bloch_diffusion_animation.gif", fps=fps,
                       phase_frames=phase_frames, dpi=dpi, **kwargs)
    return report

# Loads a saved run from disk and visualizes the forward and reverse trajectories, along with fidelity metrics and optional animations. The visualize_saved_run function reads the saved data from a specified directory, extracts the forward and reverse Bloch vectors, time arrays, and reference state information, and then calls the save_bloch_visualizations function to generate visualizations and reports based on the loaded data.
def visualize_saved_run(run_directory, **visual_options):
    run_directory = Path(run_directory)
    with np.load(run_directory / "bloch_diffusion_data.npz", allow_pickle=False) as data:
        forward = data["forward_bloch"].copy()
        reverse = data["reverse_bloch"].copy()
        times = data["forward_times"].copy()
        reverse_times = data["reverse_times"].copy()
        reference = data["reference_state"].copy() if "reference_state" in data else None
        label = str(data["reference_label"].item()) if "reference_label" in data else "Initial mixed state"
    if reference is not None and reference.size == 0:
        reference = None
    if reference is None and np.allclose(forward[:, 0], forward[0, 0], atol=1e-12, rtol=0):
        reference = density_from_bloch(forward[0, 0])
    return save_bloch_visualizations(forward, reverse, run_directory, times=times,
                                    reverse_times=reverse_times, reference_state=reference,
                                    reference_label=label, **visual_options)

# Saves visualizations of the forward and reverse trajectories, along with fidelity metrics and optional animations, to a specified output directory. The save_visualizations function takes a list of paths containing Bloch vectors and density matrices, computes the fidelity between the forward and reverse trajectories, and generates visualizations and reports based on the provided options. It also handles the creation of the output directory and saves the relevant data in compressed format.
def save_visualizations(paths, reverse_bloch, out, animate=True, *, ensemble="ball",
                        radius=.6, **visual_options):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    forward = np.stack([p["bloch"] for p in paths])
    reference, reference_label = None, "Initial mixed state"
    if np.allclose(forward[:, 0], forward[0, 0], atol=1e-12, rtol=0):
        reference = paths[0]["rho"][0]
    elif ensemble == "paper":
        reference = density_from_bloch([0., 0., radius])
        reference_label = "Initial ensemble reference direction"
    target = torch.from_numpy(forward)
    prediction = torch.from_numpy(reverse_bloch[:, ::-1].copy())
    f = fidelity(prediction, target).numpy()
    times = paths[0]["times"]
    if any(not np.array_equal(p["times"], times) for p in paths):
        raise ValueError("Visualized trajectories must share the same time grid.")
    forward_purity = (1 + (forward**2).sum(-1)) / 2
    reverse_purity = (1 + (reverse_bloch[:, ::-1]**2).sum(-1)) / 2
    np.savez_compressed(out / "bloch_diffusion_data.npz", forward_bloch=forward,
                        reverse_bloch=reverse_bloch, forward_times=times,
                        reverse_times=times[::-1], fidelity=f,
                        forward_purity=forward_purity, reverse_purity=reverse_purity,
                        reference_state=np.asarray([] if reference is None else reference),
                        reference_label=reference_label)
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    u, v = np.mgrid[0:2*np.pi:32j, 0:np.pi:16j]
    sphere = (np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v))
    ax.plot_wireframe(*sphere, color="gray", alpha=.15, linewidth=.4)
    for i in range(min(5, len(paths))):
        ax.plot(*forward[i].T, color="#176baf", alpha=.65,
                label="Forward" if i == 0 else None)
        ax.plot(*reverse_bloch[i].T, color="#e47719", linestyle="--", alpha=.8,
                label="Learned reverse" if i == 0 else None)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Mixed-state paths inside the Bloch ball")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    ax = fig.add_subplot(2, 2, 2)
    ax.fill_between(times, f.min(0), f.max(0), alpha=.2)
    ax.plot(times, f.mean(0))
    ax.set(xlabel="Forward time", ylabel="Squared Uhlmann fidelity", ylim=(0, 1.01),
           title="Paired trajectory reconstruction")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(times, forward_purity.mean(0), label="Forward")
    ax.plot(times, reverse_purity.mean(0), "--", label="Learned reverse")
    ax.set(xlabel="Forward time", ylabel="Mean trajectory purity", ylim=(.49, 1.01))
    ax.legend()
    ax = fig.add_subplot(2, 2, 4)
    for k, label in enumerate("xyz"):
        line, = ax.plot(times, forward[0, :, k], label=label)
        ax.plot(times, reverse_bloch[0, ::-1, k], "--", color=line.get_color())
    ax.set(xlabel="Forward time", ylabel="Bloch component", title="One path: solid forward, dashed reverse")
    ax.legend()
    fig.savefig(out / "mixed_state_diffusion.png", dpi=150)
    plt.close(fig)
    save_bloch_visualizations(forward, reverse_bloch, out, times=times,
                              reverse_times=times[::-1], reference_state=reference,
                              reference_label=reference_label,
                              skip_animation=not animate, **visual_options)

# Loads a checkpoint of a mixed-state diffusion RNN model from disk, ensuring that the checkpoint format is compatible. The load_checkpoint function reads the checkpoint file, verifies its format, initializes a MixedStateDiffusionRNN model with the saved configuration, loads the model's state dictionary, and sets the model to evaluation mode. It returns the loaded model and the checkpoint data for further use.
def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "mixed_qubit_record_filter_v1":
        raise ValueError("This loader requires a mixed-state filter checkpoint.")
    model = MixedStateDiffusionRNN(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

# Main function to parse command-line arguments and execute the Hamiltonian learning process for mixed-state diffusion. The main function sets up an argument parser, validates input parameters, generates training, validation, and test trajectories, and either visualizes a saved run or trains a new model based on the provided options. It handles device selection, random seed initialization, and output directory management.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble", choices=["fixed", "ball", "paper"], default="ball")
    parser.add_argument("--radius", type=float, default=.6)
    parser.add_argument("--radius-min", type=float, default=0.)
    parser.add_argument("--radius-max", type=float, default=.9)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-trajectories", type=int, default=256)
    parser.add_argument("--validation-trajectories", type=int, default=64)
    parser.add_argument("--test-trajectories", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--total-time", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    animation_group = parser.add_mutually_exclusive_group()
    animation_group.add_argument("--animate", action="store_true", dest="animate",
                                 help="Generate GIF (default; retained for compatibility).")
    animation_group.add_argument("--skip-animation", action="store_false", dest="animate",
                                 help="Save PNGs without rendering the GIF.")
    parser.set_defaults(animate=True)
    parser.add_argument("--visualize-only", type=Path, metavar="RUN_DIRECTORY",
                        help="Render saved forward/reverse arrays without retraining.")
    parser.add_argument("--viz-trajectories", type=int, default=128)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-frames", type=int, default=120,
                        help="Maximum frames per forward/reverse phase.")
    parser.add_argument("--gif-dpi", type=int, default=100)
    parser.add_argument("--snapshot-fractions", type=float, nargs="+", default=[0, .25, .5, .75, 1],
                        help="Increasing fractions of the forward duration for timestamp panels.")
    parser.add_argument("--out", type=Path, default=Path("mixed_state_results"))
    args = parser.parse_args()
    if min(args.viz_trajectories, args.gif_fps, args.gif_dpi) < 1 or args.gif_frames < 2:
        parser.error("Visualization counts, fps and dpi must be positive; gif-frames >= 2.")
    fractions = np.asarray(args.snapshot_fractions)
    if (not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1)
        or np.any(np.diff(fractions) <= 0)):
        parser.error("snapshot-fractions must increase strictly within [0,1].")
    visual_options = dict(count=args.viz_trajectories, fps=args.gif_fps,
                          phase_frames=args.gif_frames, dpi=args.gif_dpi,
                          snapshot_fractions=args.snapshot_fractions)
    if args.visualize_only is not None:
        visualize_saved_run(args.visualize_only, skip_animation=not args.animate, **visual_options)
        return
    if min(args.steps, args.epochs, args.train_trajectories, args.validation_trajectories,
           args.test_trajectories, args.batch_size, args.hidden_dim, args.threads, args.rollout_every) < 1:
        parser.error("Counts must be positive.")
    if not np.isfinite([args.gamma, args.total_time, args.lr]).all() or min(args.gamma, args.total_time, args.lr) <= 0:
        parser.error("gamma, total-time and lr must be finite and positive.")
    if not (0 <= args.radius <= 1 and 0 <= args.radius_min <= args.radius_max <= 1):
        parser.error("Radii must be in [0,1], with radius-min <= radius-max.")
    if args.gamma * args.total_time / args.steps > .25:
        parser.error("Increase --steps so gamma * total_time / steps <= 0.25 for weak steps.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    options = dict(n_steps=args.steps, gamma=args.gamma, total_time=args.total_time,
                   ensemble=args.ensemble, radius=args.radius,
                   radius_min=args.radius_min, radius_max=args.radius_max)
    train_paths, val_paths, test_paths = [generate_trajectories(n, seed=args.seed + offset, **options)
        for n, offset in [(args.train_trajectories, 1), (args.validation_trajectories, 2), (args.test_trajectories, 3)]]
    training, validation, test = [pack_trajectories(p, device) for p in [train_paths, val_paths, test_paths]]
    model = MixedStateDiffusionRNN(hidden_dim=args.hidden_dim).to(device)
    before = evaluate(model, test)
    history, best_epoch = train(model, training, validation, args.epochs, args.batch_size,
                                args.lr, args.seed, args.rollout_every)
    metrics = dict(before_training=before, after_training=evaluate(model, test), best_epoch=best_epoch,
                   task="record-conditioned mixed-state reconstruction", dtype="float64")
    example = test_paths[0]
    reverse = model.reverse_diffusion(example["rho"][-1], example["times"], example)
    exact = inverse_measurement_baseline(example)
    metrics["exact_inverse_max_matrix_error"] = float(np.max(np.abs(exact - example["rho"][::-1])))
    metrics["reverse_min_eigenvalue"] = float(np.linalg.eigvalsh(reverse["trajectory"]).min())
    if metrics["exact_inverse_max_matrix_error"] > 1e-6:
        print("WARNING: inverse baseline is numerically ill-conditioned; reduce gamma * total_time.", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format="mixed_qubit_record_filter_v1", config=model.config,
                    state_dict=model.state_dict(), rho_terminal=torch.from_numpy(example["rho"][-1]),
                    times=torch.from_numpy(example["times"]), axes=torch.from_numpy(example["axes"]),
                    outcomes=torch.from_numpy(example["outcomes"]), gamma=args.gamma), args.out / "model.pt")
    np.savez_compressed(args.out / "trajectory.npz", forward_rho=example["rho"],
                        reverse_rho=reverse["trajectory"], forward_times=example["times"],
                        reverse_times=reverse["times"], reverse_filters=reverse["filters"],
                        reverse_gains=reverse["gains"], reverse_strengths=reverse["strengths"],
                        exact_reverse_rho=exact, axes=example["axes"],
                        outcomes=example["outcomes"], gamma=args.gamma)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "metrics.json").write_text(json.dumps(dict(arguments=arguments, **metrics), indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    with torch.no_grad():
        predicted, _ = model_rollout(model, test)
    save_visualizations(test_paths, predicted.cpu().numpy(), args.out, args.animate,
                        ensemble=args.ensemble, radius=args.radius, **visual_options)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved checkpoint, density matrices, learned filters and plots to {args.out.resolve()}")


if __name__ == "__main__":
    main()
