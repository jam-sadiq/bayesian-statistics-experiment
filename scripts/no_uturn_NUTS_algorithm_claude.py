"""
A minimal No-U-Turn Sampler (NUTS) in pure NumPy.

Based on Hoffman & Gelman (2014), "The No-U-Turn Sampler", Algorithm 6
(NUTS with dual-averaging step-size adaptation), plus a diagonal mass
matrix estimated during warmup (similar to what Stan and PyMC do).

Usage
-----
You only supply a log-likelihood function of the parameters. The
priors (theta1 ~ Uniform(a, b), theta2 ~ Normal(mu, sigma)) are added
by `sample_posterior`, which also maps theta1 to an unbounded space so
NUTS never hits the hard edges of the uniform prior.

    from nuts import sample_posterior, summary

    def log_likelihood(theta):
        theta1, theta2 = theta
        return ...            # a float

    chains, info = sample_posterior(log_likelihood, a, b, mu, sigma)
    summary(chains, names=["theta1", "theta2"])

`chains` has shape (n_chains, n_samples, 2) and holds posterior draws
of (theta1, theta2) in their original units.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------
def numerical_grad(f, x, h=1e-5):
    """Central finite-difference gradient of a scalar function f at x."""
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    for i in range(x.size):
        step = h * max(1.0, abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        g[i] = (f(xp) - f(xm)) / (2.0 * step)
    return g


# ---------------------------------------------------------------------------
# Core NUTS sampler (works on any unconstrained log density)
# ---------------------------------------------------------------------------
def nuts(logp, x0, n_samples=2000, n_warmup=1000, grad=None,
         target_accept=0.8, max_depth=10, adapt_mass=True, rng=None):
    """
    Draw samples from exp(logp(x)) with NUTS.

    Parameters
    ----------
    logp : callable
        Log density (up to a constant) on an unconstrained space R^d.
    x0 : array_like
        Starting point, shape (d,).
    n_samples, n_warmup : int
        Number of kept draws and number of warmup (tuning) iterations.
    grad : callable, optional
        Gradient of logp. If None, central finite differences are used.
    target_accept : float
        Target mean acceptance statistic for step-size adaptation.
    max_depth : int
        Maximum tree depth (at most 2**max_depth leapfrog steps per draw).
    adapt_mass : bool
        Estimate a diagonal mass matrix during warmup.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict with keys
        "samples"      (n_samples, d) array of draws
        "step_size"    final step size
        "inv_metric"   final diagonal inverse mass matrix
        "accept_rate"  mean acceptance statistic after warmup
        "n_divergent"  number of divergent transitions after warmup
        "tree_depth"   (n_samples,) tree depth of each draw
    """
    rng = np.random.default_rng() if rng is None else rng
    grad = (lambda z: numerical_grad(logp, z)) if grad is None else grad
    x = np.asarray(x0, dtype=float).copy()
    d = x.size
    inv_metric = np.ones(d)          # diagonal inverse mass matrix
    DELTA_MAX = 1000.0               # energy error that counts as divergence

    def logp_and_grad(z):
        lp = logp(z)
        if lp is None or not np.isfinite(lp):
            return -np.inf, np.zeros(d)
        g = grad(z)
        if not np.all(np.isfinite(g)):
            return -np.inf, np.zeros(d)
        return float(lp), g

    def kinetic(r):
        return 0.5 * np.dot(r * inv_metric, r)

    def draw_momentum():
        return rng.standard_normal(d) / np.sqrt(inv_metric)

    def leapfrog(z, r, g, eps):
        r = r + 0.5 * eps * g
        z = z + eps * inv_metric * r
        lp, g = logp_and_grad(z)
        r = r + 0.5 * eps * g
        return z, r, g, lp

    def no_uturn(z_minus, z_plus, r_minus, r_plus):
        dz = z_plus - z_minus
        return (np.dot(dz, inv_metric * r_minus) >= 0 and
                np.dot(dz, inv_metric * r_plus) >= 0)

    def find_reasonable_eps(z, lp, g):
        eps = 1.0
        r = draw_momentum()
        h0 = lp - kinetic(r)
        _, r1, _, lp1 = leapfrog(z, r, g, eps)
        log_ratio = (lp1 - kinetic(r1)) - h0 if np.isfinite(lp1) else -np.inf
        direction = 1.0 if log_ratio > np.log(0.5) else -1.0
        while direction * log_ratio > -direction * np.log(2.0):
            eps *= 2.0 ** direction
            if eps < 1e-10 or eps > 1e10:
                break
            _, r1, _, lp1 = leapfrog(z, r, g, eps)
            log_ratio = (lp1 - kinetic(r1)) - h0 if np.isfinite(lp1) else -np.inf
        return eps

    state = {"divergent": False}

    def build_tree(z, r, g, log_u, v, j, eps, h0):
        """Recursively build a subtree of 2**j leapfrog steps in direction v."""
        if j == 0:
            z1, r1, g1, lp1 = leapfrog(z, r, g, v * eps)
            h1 = lp1 - kinetic(r1) if np.isfinite(lp1) else -np.inf
            n1 = int(log_u <= h1)
            s1 = int(log_u < h1 + DELTA_MAX)
            if not s1:
                state["divergent"] = True
            alpha = np.exp(min(0.0, h1 - h0)) if np.isfinite(h1) else 0.0
            return z1, r1, g1, z1, r1, g1, z1, g1, lp1, n1, s1, alpha, 1

        (z_m, r_m, g_m, z_p, r_p, g_p, z_prop, g_prop, lp_prop,
         n1, s1, alpha1, n_alpha1) = build_tree(z, r, g, log_u, v, j - 1, eps, h0)

        if s1:
            if v == -1:
                (z_m, r_m, g_m, _, _, _, z_prop2, g_prop2, lp_prop2,
                 n2, s2, alpha2, n_alpha2) = build_tree(z_m, r_m, g_m, log_u, v, j - 1, eps, h0)
            else:
                (_, _, _, z_p, r_p, g_p, z_prop2, g_prop2, lp_prop2,
                 n2, s2, alpha2, n_alpha2) = build_tree(z_p, r_p, g_p, log_u, v, j - 1, eps, h0)
            if n1 + n2 > 0 and rng.uniform() < n2 / (n1 + n2):
                z_prop, g_prop, lp_prop = z_prop2, g_prop2, lp_prop2
            alpha1 += alpha2
            n_alpha1 += n_alpha2
            s1 = int(s2 and no_uturn(z_m, z_p, r_m, r_p))
            n1 += n2

        return (z_m, r_m, g_m, z_p, r_p, g_p, z_prop, g_prop, lp_prop,
                n1, s1, alpha1, n_alpha1)

    # --- initialisation -----------------------------------------------------
    lp, g = logp_and_grad(x)
    if not np.isfinite(lp):
        raise ValueError("logp is not finite at the starting point x0.")

    def init_dual_averaging(eps):
        return {"mu": np.log(10.0 * eps), "H_bar": 0.0,
                "log_eps_bar": 0.0, "t": 0}

    eps = find_reasonable_eps(x, lp, g)
    da = init_dual_averaging(eps)
    gamma, t0, kappa = 0.05, 10.0, 0.75

    # Warmup schedule: step size only, then collect draws for the mass
    # matrix, then re-tune the step size with the new mass matrix.
    window_start = int(0.15 * n_warmup)
    window_end = int(0.85 * n_warmup)
    use_window = adapt_mass and n_warmup >= 100
    window_draws = []

    samples = np.empty((n_samples, d))
    depths = np.empty(n_samples, dtype=int)
    accept_stats = []
    n_divergent = 0

    for m in range(n_warmup + n_samples):
        # --- one NUTS transition -------------------------------------------
        r0 = draw_momentum()
        h0 = lp - kinetic(r0)
        log_u = h0 - rng.exponential()          # log of slice variable

        z_m = z_p = x
        r_m = r_p = r0
        g_m = g_p = g
        j, n, s = 0, 1, 1
        alpha, n_alpha = 0.0, 1
        state["divergent"] = False

        while s and j < max_depth:
            v = -1 if rng.uniform() < 0.5 else 1
            if v == -1:
                (z_m, r_m, g_m, _, _, _, z_prop, g_prop, lp_prop,
                 n1, s1, alpha, n_alpha) = build_tree(z_m, r_m, g_m, log_u, v, j, eps, h0)
            else:
                (_, _, _, z_p, r_p, g_p, z_prop, g_prop, lp_prop,
                 n1, s1, alpha, n_alpha) = build_tree(z_p, r_p, g_p, log_u, v, j, eps, h0)
            if s1 and rng.uniform() < min(1.0, n1 / n):
                x, g, lp = z_prop, g_prop, lp_prop
            n += n1
            s = int(s1 and no_uturn(z_m, z_p, r_m, r_p))
            j += 1

        accept = alpha / n_alpha

        # --- adaptation during warmup --------------------------------------
        if m < n_warmup:
            da["t"] += 1
            t = da["t"]
            w = 1.0 / (t + t0)
            da["H_bar"] = (1 - w) * da["H_bar"] + w * (target_accept - accept)
            log_eps = da["mu"] - np.sqrt(t) / gamma * da["H_bar"]
            eta = t ** (-kappa)
            da["log_eps_bar"] = eta * log_eps + (1 - eta) * da["log_eps_bar"]
            eps = np.exp(log_eps)

            if use_window and window_start <= m < window_end:
                window_draws.append(x.copy())
            if use_window and m == window_end - 1:
                w_draws = np.array(window_draws)
                k = len(w_draws)
                var = w_draws.var(axis=0, ddof=1)
                # Shrink towards 1e-3 like Stan, for stability
                inv_metric = (k / (k + 5.0)) * var + 1e-3 * (5.0 / (k + 5.0))
                eps = find_reasonable_eps(x, lp, g)
                da = init_dual_averaging(eps)

            if m == n_warmup - 1:
                eps = np.exp(da["log_eps_bar"])   # freeze step size
        else:
            i = m - n_warmup
            samples[i] = x
            depths[i] = j
            accept_stats.append(accept)
            n_divergent += int(state["divergent"])

    return {
        "samples": samples,
        "step_size": eps,
        "inv_metric": inv_metric,
        "accept_rate": float(np.mean(accept_stats)) if accept_stats else np.nan,
        "n_divergent": n_divergent,
        "tree_depth": depths,
    }


# ---------------------------------------------------------------------------
# Posterior for your model: theta1 ~ U(a, b), theta2 ~ N(mu, sigma)
# ---------------------------------------------------------------------------
def sample_posterior(log_likelihood, a, b, mu, sigma, n_chains=4,
                     n_samples=2000, n_warmup=1000, target_accept=0.8,
                     seed=None, verbose=True):
    """
    Sample the posterior of (theta1, theta2) with NUTS.

    Prior: theta1 ~ Uniform(a, b), theta2 ~ Normal(mu, sigma).
    theta1 is sampled through z1 = logit((theta1 - a) / (b - a)), with the
    Jacobian term added, so the sampler works on an unbounded space.

    Parameters
    ----------
    log_likelihood : callable
        log_likelihood(theta) -> float, where theta = [theta1, theta2].
        Return only the log-likelihood; the log prior is added here.
    a, b : float
        Limits of the uniform prior on theta1.
    mu, sigma : float
        Mean and standard deviation of the Gaussian prior on theta2.
    n_chains, n_samples, n_warmup : int
    target_accept : float
    seed : int, optional

    Returns
    -------
    chains : ndarray, shape (n_chains, n_samples, 2)
        Posterior draws of (theta1, theta2).
    info : list of dict
        Per-chain sampler diagnostics (see `nuts`).
    """
    rng = np.random.default_rng(seed)
    log_width = np.log(b - a)

    def to_theta(z):
        s = np.exp(-np.logaddexp(0.0, -z[0]))   # stable sigmoid(z1)
        return np.array([a + (b - a) * s, z[1]])

    def log_post(z):
        theta = to_theta(z)
        ll = log_likelihood(theta)
        if ll is None or not np.isfinite(ll):
            return -np.inf
        log_prior_u = -log_width
        log_prior_n = (-0.5 * np.log(2 * np.pi * sigma ** 2)
                       - (theta[1] - mu) ** 2 / (2 * sigma ** 2))
        # log |d theta1 / d z1| = log(b-a) + log(s) + log(1-s)
        log_jac = log_width - np.logaddexp(0.0, -z[0]) - np.logaddexp(0.0, z[0])
        return ll + log_prior_u + log_prior_n + log_jac

    chains = np.empty((n_chains, n_samples, 2))
    info = []
    for c in range(n_chains):
        z0 = np.array([rng.uniform(-2, 2), mu + sigma * rng.uniform(-1, 1)])
        res = nuts(log_post, z0, n_samples=n_samples, n_warmup=n_warmup,
                   target_accept=target_accept, rng=rng)
        chains[c] = np.array([to_theta(z) for z in res["samples"]])
        info.append(res)
        if verbose:
            print(f"chain {c + 1}/{n_chains}: step size {res['step_size']:.3f}, "
                  f"accept {res['accept_rate']:.2f}, "
                  f"divergences {res['n_divergent']}, "
                  f"mean tree depth {res['tree_depth'].mean():.1f}")
    return chains, info


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def split_rhat(x):
    """Split R-hat for draws of one parameter, x shape (chains, draws)."""
    half = x.shape[1] // 2
    s = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    n = s.shape[1]
    W = s.var(axis=1, ddof=1).mean()
    B = n * s.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def summary(chains, names=None):
    """Print mean, sd, 95% interval and R-hat for each parameter."""
    n_par = chains.shape[2]
    names = names or [f"theta{i + 1}" for i in range(n_par)]
    print(f"{'param':>8} {'mean':>9} {'sd':>8} {'2.5%':>9} {'50%':>9} {'97.5%':>9} {'r_hat':>6}")
    for i, name in enumerate(names):
        x = chains[:, :, i]
        flat = x.ravel()
        q = np.percentile(flat, [2.5, 50, 97.5])
        print(f"{name:>8} {flat.mean():9.4f} {flat.std():8.4f} "
              f"{q[0]:9.4f} {q[1]:9.4f} {q[2]:9.4f} {split_rhat(x):6.3f}")


# ---------------------------------------------------------------------------
