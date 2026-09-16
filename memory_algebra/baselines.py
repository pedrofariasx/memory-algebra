from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LSTMCell:
    def __init__(self, dim: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(hidden)
        self.Wf = rng.standard_normal((hidden, dim + hidden)) * scale
        self.Wi = rng.standard_normal((hidden, dim + hidden)) * scale
        self.Wo = rng.standard_normal((hidden, dim + hidden)) * scale
        self.Wg = rng.standard_normal((hidden, dim + hidden)) * scale
        self.bf = np.ones(hidden)
        self.h = np.zeros(hidden)
        self.c = np.zeros(hidden)

    def step(self, x: np.ndarray) -> np.ndarray:
        z = np.concatenate([np.asarray(x, dtype=float), self.h])
        f = _sigmoid(self.Wf @ z + self.bf)
        i = _sigmoid(self.Wi @ z)
        o = _sigmoid(self.Wo @ z)
        g = np.tanh(self.Wg @ z)
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        return self.h

    def run_sequence(self, xs: list[np.ndarray]) -> np.ndarray:
        self.h = np.zeros(self.h.shape)
        self.c = np.zeros(self.c.shape)
        for x in xs:
            self.step(x)
        return self.h


def train_lstm_recall(
    dim: int,
    hidden: int,
    sequences: list[list[np.ndarray]],
    steps: int = 500,
    lr: float = 1e-3,
    seed: int = 11,
) -> LSTMCell:
    cell = LSTMCell(dim=dim, hidden=hidden, seed=seed)
    params = [cell.Wf, cell.Wi, cell.Wo, cell.Wg, cell.bf]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 1)

    for t in range(steps):
        seq = sequences[rng.integers(len(sequences))]
        target = np.asarray(seq[0], dtype=float)
        cell.Wf, cell.Wi, cell.Wo, cell.Wg, cell.bf = params
        h = cell.run_sequence(seq)
        hn = float(np.linalg.norm(h))
        tn = float(np.linalg.norm(target))
        if hn < 1e-12 or tn < 1e-12:
            continue
        loss_grad_h = -(target - h * (np.dot(h, target) / (hn * tn)) / (hn * tn))
        grads = _bptt(cell, seq, loss_grad_h)
        for k in range(len(params)):
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            m_hat = m[k] / (1 - beta1 ** (t + 1))
            v_hat = v[k] / (1 - beta2 ** (t + 1))
            params[k] = params[k] - lr * m_hat / (np.sqrt(v_hat) + eps)

    cell.Wf, cell.Wi, cell.Wo, cell.Wg, cell.bf = params
    return cell


def _bptt(cell: LSTMCell, xs: list[np.ndarray], grad_h: np.ndarray) -> list[np.ndarray]:
    h_dim = cell.h.shape[0]
    d_total = cell.Wf.shape[1]
    xs = [np.asarray(x, dtype=float) for x in xs]
    T = len(xs)

    hs = [np.zeros(h_dim)]
    cs = [np.zeros(h_dim)]
    fs, iss, os_, gs, zs = [], [], [], [], []
    h_prev = np.zeros(h_dim)
    c_prev = np.zeros(h_dim)
    for x in xs:
        z = np.concatenate([x, h_prev])
        f = _sigmoid(cell.Wf @ z + cell.bf)
        i = _sigmoid(cell.Wi @ z)
        o = _sigmoid(cell.Wo @ z)
        g = np.tanh(cell.Wg @ z)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        zs.append(z)
        fs.append(f)
        iss.append(i)
        os_.append(o)
        gs.append(g)
        hs.append(h)
        cs.append(c)
        h_prev = h
        c_prev = c

    dWf = np.zeros_like(cell.Wf)
    dWi = np.zeros_like(cell.Wi)
    dWo = np.zeros_like(cell.Wo)
    dWg = np.zeros_like(cell.Wg)
    dbf = np.zeros_like(cell.bf)

    dh = grad_h.copy()
    dc = np.zeros(h_dim)
    for t in range(T - 1, -1, -1):
        z = zs[t]
        tanh_c = np.tanh(cs[t + 1])
        do = dh * tanh_c
        dc += dh * os_[t] * (1 - tanh_c**2)
        df = dc * cs[t]
        di = dc * gs[t]
        dg = dc * iss[t]
        dc_prev = dc * fs[t]

        dz_f = df * fs[t] * (1 - fs[t])
        dz_i = di * iss[t] * (1 - iss[t])
        dz_o = do * os_[t] * (1 - os_[t])
        dz_g = dg * (1 - gs[t] ** 2)

        dWf += np.outer(dz_f, z)
        dWi += np.outer(dz_i, z)
        dWo += np.outer(dz_o, z)
        dWg += np.outer(dz_g, z)
        dbf += dz_f

        dz = cell.Wf.T @ dz_f + cell.Wi.T @ dz_i + cell.Wo.T @ dz_o + cell.Wg.T @ dz_g
        dh = dz[h_dim:]
        dc = dc_prev

    return [dWf, dWi, dWo, dWg, dbf]
