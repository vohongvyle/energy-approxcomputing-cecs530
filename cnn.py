"""
A small CNN in plain NumPy, plus its 8-bit lookup-table version.

    conv 3x3 (16 filters, padding 1) -> ReLU -> max-pool 2x2
    conv 3x3 (32 filters, padding 1) -> ReLU -> max-pool 2x2
    dense -> classes

Convolutions are written as im2col + matrix multiply, so the 8-bit version can send every
product through a multiplier's lookup table with fastlut.lut_matmul, exactly like the MLP.
"""
import numpy as np
from fastlut import lut_matmul


def im2col(x):
    """(N, H, W, C) -> (N, H, W, 9C): the 3x3 neighbourhood of every pixel, zero padding 1."""
    N, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return np.concatenate([xp[:, dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)], axis=-1)


def col2im(dcols, C):
    """Gradient of im2col: add each of the 9 shifted slices back."""
    N, H, W, _ = dcols.shape
    dxp = np.zeros((N, H + 2, W + 2, C), dcols.dtype)
    k = 0
    for dy in range(3):
        for dx in range(3):
            dxp[:, dy:dy + H, dx:dx + W, :] += dcols[..., k * C:(k + 1) * C]
            k += 1
    return dxp[:, 1:-1, 1:-1, :]


def pool(a):
    N, H, W, C = a.shape
    return a.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def pool_back(d, a, m):
    """Route the gradient to the position that held the maximum."""
    N, H, W, C = a.shape
    up = lambda t: t.repeat(2, axis=1).repeat(2, axis=2)
    mask = (a == up(m))
    return mask * up(d)


class CNN:
    def __init__(self, shape, n_classes, seed=0, c1=16, c2=32):
        H, W, C = shape
        rng = np.random.default_rng(seed)
        he = lambda fan_in, fan_out: (rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))).astype(np.float32)
        self.shape, self.c1, self.c2 = shape, c1, c2
        self.p = {"W1": he(9 * C, c1), "b1": np.zeros(c1, np.float32),
                  "W2": he(9 * c1, c2), "b2": np.zeros(c2, np.float32),
                  "W3": he((H // 4) * (W // 4) * c2, n_classes), "b3": np.zeros(n_classes, np.float32)}

    def forward(self, x, keep=False):
        p = self.p
        P1 = im2col(x); Z1 = P1 @ p["W1"] + p["b1"]; A1 = np.maximum(Z1, 0); M1 = pool(A1)
        P2 = im2col(M1); Z2 = P2 @ p["W2"] + p["b2"]; A2 = np.maximum(Z2, 0); M2 = pool(A2)
        F = M2.reshape(len(x), -1); Z3 = F @ p["W3"] + p["b3"]
        return (Z3, dict(P1=P1, Z1=Z1, A1=A1, M1=M1, P2=P2, Z2=Z2, A2=A2, M2=M2, F=F)) if keep else Z3

    def backward(self, dZ3, c):
        p, g = self.p, {}
        g["W3"] = c["F"].T @ dZ3; g["b3"] = dZ3.sum(0)
        dM2 = (dZ3 @ p["W3"].T).reshape(c["M2"].shape)
        dZ2 = pool_back(dM2, c["A2"], c["M2"]) * (c["Z2"] > 0)
        g["W2"] = c["P2"].reshape(-1, c["P2"].shape[-1]).T @ dZ2.reshape(-1, self.c2); g["b2"] = dZ2.sum((0, 1, 2))
        dM1 = col2im(dZ2 @ p["W2"].T, self.c1)
        dZ1 = pool_back(dM1, c["A1"], c["M1"]) * (c["Z1"] > 0)
        g["W1"] = c["P1"].reshape(-1, c["P1"].shape[-1]).T @ dZ1.reshape(-1, self.c1); g["b1"] = dZ1.sum((0, 1, 2))
        return g

    def fit(self, X, y, epochs, lr=1e-3, batch=128, wd=1e-4, seed=0, log=None):
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        t = 0
        for ep in range(epochs):
            order = rng.permutation(len(X)); loss = 0.0
            for i in range(0, len(X), batch):
                b = order[i:i + batch]
                z, c = self.forward(X[b], keep=True)
                z = z - z.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
                loss -= np.log(pr[np.arange(len(b)), y[b]] + 1e-9).sum()
                d = pr; d[np.arange(len(b)), y[b]] -= 1; d /= len(b)
                g = self.backward(d.astype(np.float32), c)
                t += 1
                for k in self.p:
                    gk = g[k] + (wd * self.p[k] if k.startswith("W") else 0)
                    m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if log:
                log(f"  epoch {ep + 1}/{epochs}  loss {loss / len(X):.4f}")
        return self

    def predict_logits(self, X, batch=1000):
        return np.concatenate([self.forward(X[i:i + batch]) for i in range(0, len(X), batch)])


class QuantCNN:
    """8-bit version: unsigned 8-bit activations, weights as sign + 8-bit magnitude, every product
    read from a multiplier's lookup table."""

    def __init__(self, cnn, Xcal):
        self.cnn = cnn
        p = cnn.p
        _, c = cnn.forward(Xcal, keep=True)
        ins = [c["P1"], c["P2"], c["F"]]
        self.layers = []
        for (W, b), a in zip(((p["W1"], p["b1"]), (p["W2"], p["b2"]), (p["W3"], p["b3"])), ins):
            sa = max(float(np.percentile(a, 99.9)), 1e-8) / 255.0
            sw = float(np.abs(W).max()) / 255.0
            q = np.clip(np.round(W / sw), -255, 255)
            self.layers.append(dict(sa=sa, sw=sw, sgn=np.sign(q).astype(np.int8), mag=np.abs(q).astype(np.uint8),
                                    b=b.astype(np.float64)))
        H, W, _ = cnn.shape
        # multiplications per image, per layer
        self.macs = [H * W * self.layers[0]["mag"].size, (H // 2) * (W // 2) * self.layers[1]["mag"].size,
                     self.layers[2]["mag"].size]

    def _quant(self, k, a):
        return np.clip(np.round(a / self.layers[k]["sa"]), 0, 255).astype(np.uint8)

    def _mm(self, k, xq, lut):
        L = self.layers[k]
        return lut_matmul(xq, L["mag"], L["sgn"], lut) * np.float32(L["sa"] * L["sw"]) + L["b"].astype(np.float32)

    def prepare(self, X, batch=1000):
        """Layer-1 input columns, quantized once: they do not depend on the multiplier.
        Quantizing before im2col gives the same result, because im2col only copies values."""
        return [im2col(self._quant(0, X[i:i + batch])) for i in range(0, len(X), batch)]

    def forward(self, prepared, luts):
        """prepared: output of prepare(); luts: one 256x256 table per layer (3 layers)."""
        out = []
        for c1 in prepared:
            N, H, W, K = c1.shape
            A1 = np.maximum(self._mm(0, c1.reshape(-1, K), luts[0]), 0).reshape(N, H, W, -1)
            c2 = im2col(self._quant(1, pool(A1)))
            A2 = np.maximum(self._mm(1, c2.reshape(-1, c2.shape[-1]), luts[1]), 0).reshape(N, H // 2, W // 2, -1)
            out.append(self._mm(2, self._quant(2, pool(A2).reshape(N, -1)), luts[2]))
        return np.concatenate(out)


def grad_check(seed=0):
    """Compare backward() with finite differences on a tiny random problem."""
    rng = np.random.default_rng(seed)
    net = CNN((8, 8, 2), 3, seed=seed, c1=4, c2=5)
    for k in net.p:
        net.p[k] = net.p[k].astype(np.float64) + (0.1 * rng.normal(size=net.p[k].shape) if k.startswith("b") else 0)
    x = rng.random((4, 8, 8, 2)); y = rng.integers(0, 3, 4)

    def loss():
        z = net.forward(x); z = z - z.max(1, keepdims=True)
        return -(z[np.arange(4), y] - np.log(np.exp(z).sum(1))).mean()

    z, c = net.forward(x, keep=True)
    z = z - z.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    d = pr.copy(); d[np.arange(4), y] -= 1; d /= 4
    g = net.backward(d, c)
    worst = 0.0
    for k in net.p:
        for idx in [tuple(rng.integers(0, s) for s in net.p[k].shape) for _ in range(5)]:
            old = net.p[k][idx]
            net.p[k][idx] = old + 1e-6; lp = loss()
            net.p[k][idx] = old - 1e-6; lm = loss()
            net.p[k][idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
    return worst


if __name__ == "__main__":
    print(f"gradient check, worst relative error: {grad_check():.2e}")
