"""Small numpy tools for the theory text, chapters 10 and 11.

statract has no penalized regression or tree ensembles yet, so the text uses
these short, readable versions. They are for teaching, not for production.

- ``logistic_path``: ridge or LASSO logistic regression along a lambda path
  (IRLS outer loop; ridge solves each step exactly, LASSO uses coordinate
  descent as in glmnet).
- ``cart`` / ``predict_cart``: a regression tree on a numeric matrix.
- ``random_forest`` and ``boosting``: ensembles of those trees.
"""

from __future__ import annotations

import numpy as np


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mean) / sd, mean, sd


def lambda_max(xs: np.ndarray, y: np.ndarray) -> float:
    """Smallest lambda at which every LASSO coefficient is zero."""
    return float(np.max(np.abs(xs.T @ (y - y.mean()))) / len(y))


def logistic_path(
    xs: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    *,
    penalty: str,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalized logistic regression on standardized ``xs`` for each lambda.

    Minimizes  -loglik / n + lambda * P(beta), with P = sum |beta| (LASSO) or
    sum beta^2 / 2 (ridge). The intercept is not penalized. Returns the
    intercepts (len(lambdas),) and coefficients (len(lambdas), p).
    """
    n, p = xs.shape
    b0 = float(np.log(y.mean() / (1 - y.mean())))
    beta = np.zeros(p)
    b0s, betas = [], []
    for lam in lambdas:
        for _ in range(max_iter):
            eta = b0 + xs @ beta
            prob = np.clip(_expit(eta), 1e-6, 1 - 1e-6)
            w = prob * (1 - prob)
            z = eta + (y - prob) / w
            old = np.r_[b0, beta]
            if penalty == "ridge":
                # Weighted least squares with an unpenalized intercept, solved exactly.
                xa = np.c_[np.ones(n), xs]
                a = xa.T @ (w[:, None] * xa) / n
                a[1:, 1:] += lam * np.eye(p)
                sol = np.linalg.solve(a, xa.T @ (w * z) / n)
                b0, beta = float(sol[0]), sol[1:].copy()
            else:
                xw2 = (w[:, None] * xs**2).sum(axis=0) / n
                for _sweep in range(200):
                    prev = beta.copy()
                    r = z - b0 - xs @ beta
                    b0 += float((w * r).sum() / w.sum())
                    r = z - b0 - xs @ beta
                    for j in range(p):
                        rho = float((w * xs[:, j] * r).sum() / n) + xw2[j] * beta[j]
                        new = np.sign(rho) * max(abs(rho) - lam, 0.0) / xw2[j]
                        r += xs[:, j] * (beta[j] - new)
                        beta[j] = new
                    if np.max(np.abs(beta - prev)) < tol:
                        break
            if np.max(np.abs(np.r_[b0, beta] - old)) < tol:
                break
        if penalty != "ridge":
            beta[np.abs(beta) < 1e-10] = 0.0  # rounding at the edge of the path
        b0s.append(b0)
        betas.append(beta.copy())
    return np.array(b0s), np.array(betas)


def cv_deviance(
    xs: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    *,
    penalty: str,
    k: int = 10,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of held-out deviance per patient, k-fold CV."""
    folds = np.random.default_rng(seed).permutation(np.arange(len(y)) % k)
    dev = np.zeros((k, len(lambdas)))
    for f in range(k):
        tr, te = folds != f, folds == f
        b0, beta = logistic_path(xs[tr], y[tr], lambdas, penalty=penalty)
        prob = np.clip(_expit(b0[:, None] + beta @ xs[te].T), 1e-12, 1 - 1e-12)
        dev[f] = -2 * np.mean(
            y[te] * np.log(prob) + (1 - y[te]) * np.log(1 - prob), axis=1
        )
    return dev.mean(axis=0), dev.std(axis=0, ddof=1) / np.sqrt(k)


# ---------------------------------------------------------------------------
# Trees


def cart(
    x: np.ndarray,
    g: np.ndarray,
    *,
    max_depth: int,
    min_leaf: int = 20,
    features: int | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    """Least-squares regression tree for target ``g``.

    Nodes are lists: ["leaf", value] or ["split", j, threshold, left, right].
    ``features`` picks that many random columns at each split (random forest).
    """

    def grow(idx: np.ndarray, depth: int) -> list:
        if depth == max_depth or len(idx) < 2 * min_leaf:
            return ["leaf", float(g[idx].mean())]
        cols = np.arange(x.shape[1])
        if features is not None and rng is not None:
            cols = rng.choice(cols, size=features, replace=False)
        best = None
        total, m = g[idx].sum(), len(idx)
        for j in cols:
            order = idx[np.argsort(x[idx, j], kind="stable")]
            xv, gv = x[order, j], g[order]
            cum = np.cumsum(gv)
            left_n = np.arange(1, m)
            ok = (left_n >= min_leaf) & (m - left_n >= min_leaf) & (xv[1:] > xv[:-1])
            if not ok.any():
                continue
            ls = cum[:-1]
            rs = total - ls
            # Reduction in the sum of squares from splitting here.
            gain = ls**2 / left_n + rs**2 / (m - left_n) - total**2 / m
            gain[~ok] = -np.inf
            i = int(np.argmax(gain))
            if best is None or gain[i] > best[0]:
                best = (gain[i], j, (xv[i] + xv[i + 1]) / 2)
        if best is None or best[0] <= 1e-12:
            return ["leaf", float(g[idx].mean())]
        _, j, t = best
        left = idx[x[idx, j] <= t]
        right = idx[x[idx, j] > t]
        return [
            "split",
            int(j),
            float(t),
            grow(left, depth + 1),
            grow(right, depth + 1),
        ]

    return grow(np.arange(len(g)), 0)


def predict_cart(tree: list, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x))

    def walk(node: list, idx: np.ndarray) -> None:
        if node[0] == "leaf":
            out[idx] = node[1]
            return
        _, j, t, left, right = node
        mask = x[idx, j] <= t
        walk(left, idx[mask])
        walk(right, idx[~mask])

    walk(tree, np.arange(len(x)))
    return out


def random_forest(
    x: np.ndarray,
    y: np.ndarray,
    *,
    trees: int = 200,
    max_depth: int = 8,
    min_leaf: int = 10,
    features: int | None = None,
    seed: int = 1,
) -> list:
    """Bagged regression trees on 0/1 ``y``; the mean of leaves is a risk."""
    rng = np.random.default_rng(seed)
    n, p = x.shape
    features = features or max(1, int(np.sqrt(p)))
    forest = []
    for _ in range(trees):
        boot = rng.integers(0, n, n)
        forest.append(
            cart(
                x[boot],
                y[boot].astype(float),
                max_depth=max_depth,
                min_leaf=min_leaf,
                features=features,
                rng=rng,
            )
        )
    return forest


def predict_forest(forest: list, x: np.ndarray) -> np.ndarray:
    return np.mean([predict_cart(t, x) for t in forest], axis=0)


def _newton_leaves(
    node: list, x: np.ndarray, idx: np.ndarray, grad: np.ndarray, hess: np.ndarray
) -> None:
    """Replace each leaf value with the Newton step sum(y - p) / sum(p (1 - p))."""
    if node[0] == "leaf":
        node[1] = float(grad[idx].sum() / max(hess[idx].sum(), 1e-12))
        return
    _, j, t, left, right = node
    mask = x[idx, j] <= t
    _newton_leaves(left, x, idx[mask], grad, hess)
    _newton_leaves(right, x, idx[~mask], grad, hess)


def boosting(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rounds: int,
    depth: int = 2,
    rate: float = 0.1,
    min_leaf: int = 20,
) -> tuple[float, float, list]:
    """Gradient boosting for log loss: each small tree fits the residual y - p."""
    f0 = float(np.log(y.mean() / (1 - y.mean())))
    eta = np.full(len(y), f0)
    stages = []
    for _ in range(rounds):
        prob = _expit(eta)
        resid = y - prob
        tree = cart(x, resid, max_depth=depth, min_leaf=min_leaf)
        _newton_leaves(tree, x, np.arange(len(y)), resid, prob * (1 - prob))
        eta += rate * predict_cart(tree, x)
        stages.append(tree)
    return f0, rate, stages


def predict_boosting(
    model: tuple[float, float, list], x: np.ndarray, *, rounds: int | None = None
) -> np.ndarray:
    """Predicted risk after ``rounds`` trees (all trees when None)."""
    f0, rate, stages = model
    eta = np.full(len(x), f0)
    for tree in stages[: rounds if rounds is not None else len(stages)]:
        eta += rate * predict_cart(tree, x)
    return _expit(eta)


# ---------------------------------------------------------------------------
# A one-hidden-layer neural network (chapter 12)


def mlp_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    decay: float = 0.0,
    epochs: int = 3000,
    rate: float = 0.01,
    seed: int = 1,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict:
    """Logistic units stacked in two layers, trained by full-batch Adam on log loss.

    hidden layer: h = tanh(x W1 + b1); output: p = expit(h w2 + b2).
    ``decay`` is an L2 penalty on the weights (the ridge penalty of chapter 10).
    With ``x_val``/``y_val`` the held-out log loss is recorded every epoch.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    params = {
        "w1": rng.normal(0, 1 / np.sqrt(d), (d, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden),
        "b2": np.array(float(np.log(y.mean() / (1 - y.mean())))),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    history = []
    for t in range(1, epochs + 1):
        h = np.tanh(x @ params["w1"] + params["b1"])
        p = _expit(h @ params["w2"] + params["b2"])
        g_out = (p - y) / n
        grads = {
            "w2": h.T @ g_out + decay * params["w2"],
            "b2": np.array(g_out.sum()),
        }
        g_h = np.outer(g_out, params["w2"]) * (1 - h**2)
        grads["w1"] = x.T @ g_h + decay * params["w1"]
        grads["b1"] = g_h.sum(axis=0)
        for k, value in params.items():
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            mh = m[k] / (1 - 0.9**t)
            vh = v[k] / (1 - 0.999**t)
            params[k] = value - rate * mh / (np.sqrt(vh) + 1e-8)
        if x_val is not None and y_val is not None and (t % 10 == 0 or t == 1):
            pv = np.clip(mlp_predict(params, x_val), 1e-12, 1 - 1e-12)
            pt = np.clip(p, 1e-12, 1 - 1e-12)
            history.append(
                (
                    t,
                    float(-np.mean(y * np.log(pt) + (1 - y) * np.log(1 - pt))),
                    float(-np.mean(y_val * np.log(pv) + (1 - y_val) * np.log(1 - pv))),
                )
            )
    params["history"] = history
    return params


def mlp_predict(params: dict, x: np.ndarray) -> np.ndarray:
    h = np.tanh(x @ params["w1"] + params["b1"])
    return _expit(h @ params["w2"] + params["b2"])


def newton_logistic(x: np.ndarray, y: np.ndarray, iters: int = 8) -> list[np.ndarray]:
    """Plain Newton-Raphson for the logistic log likelihood, from all zeros."""
    beta = np.zeros(x.shape[1])
    history = [beta.copy()]
    for _ in range(iters):
        p = _expit(x @ beta)
        info = x.T @ (x * (p * (1 - p))[:, None])
        beta = beta + np.linalg.solve(info, x.T @ (y - p))
        history.append(beta.copy())
    return history


def firth_logistic(
    x: np.ndarray, y: np.ndarray, tol: float = 1e-8, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Firth's penalized likelihood (Jeffreys prior) for logistic regression.

    Returns the estimates and their Wald standard errors. The score gets the
    extra term ``h_i (1/2 - p_i)``, where ``h`` is the hat diagonal.
    """
    beta = np.zeros(x.shape[1])
    for _ in range(max_iter):
        p = _expit(x @ beta)
        w = p * (1 - p)
        inv = np.linalg.inv(x.T @ (x * w[:, None]))
        h = w * np.sum((x @ inv) * x, axis=1)
        step = inv @ (x.T @ (y - p + h * (0.5 - p)))
        step = np.clip(step, -5, 5)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    p = _expit(x @ beta)
    inv = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    return beta, np.sqrt(np.diag(inv))
