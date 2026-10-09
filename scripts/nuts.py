"""
A minimal No-U-Turn Sampler (NUTS) in pure NumPy, with convergence
diagnostics and plots.

Based on Hoffman & Gelman (2014), "The No-U-Turn Sampler", Algorithm 6
(NUTS with dual-averaging step-size adaptation), plus Stan-style windowed
adaptation of a diagonal or dense mass matrix during warmup.
Diagnostics (rank-normalised split R-hat, bulk and tail ESS) follow
Vehtari et al. (2021), the same definitions used by Stan and ArviZ.

Usage
-----
You supply a log-likelihood and one prior per parameter. The sampler adds
the log prior and maps bounded parameters to an unbounded space, so NUTS
never hits the hard edges of a uniform prior.

    from nuts import sample, Uniform, Normal, summary, check_convergence
    from nuts import trace_plot, corner_plot

    def log_likelihood(theta, data):
        H0, MB = theta
        return ...            # a float

    priors = [Uniform(50, 100), Normal(-19.2, 0.12)]
    chains, info = sample(log_likelihood, priors, args=(data,))

    summary(chains, names=["H0", "MB"])
    check_convergence(chains, info)
    trace_plot(chains, info, names=["H0", "MB"], filename="trace.png")
    corner_plot(chains, names=["H0", "MB"], filename="corner.png")

`chains` has shape (n_chains, n_samples, n_params) and holds posterior
draws in their original units. `info` is a list of per-chain sampler
statistics (step size, divergences, tree depth, ...).

The older call `sample_posterior(log_likelihood, a, b, mu, sigma)`
still works: it is `sample` with priors [Uniform(a, b), Normal(mu, sigma)].
"""

import numpy as np


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
def _softplus(x):
    """log(1 + exp(x)), computed without overflow."""
    return np.logaddexp(0.0, x)


class Uniform:
    """
    Uniform(lower, upper) prior.

        log p(x) = -log(upper - lower)   if lower <= x <= upper
                 = -inf                  otherwise

    Sampled through z = logit((x - lower) / (upper - lower)), which maps
    the interval onto the whole real line.
    """

    def __init__(self, lower, upper):
        if not upper > lower:
            raise ValueError("Uniform prior needs upper > lower.")
        self.lower, self.upper = float(lower), float(upper)
        self.width = self.upper - self.lower

    def logpdf(self, x):
        if self.lower <= x <= self.upper:
            return -np.log(self.width)
        return -np.inf

    def from_unconstrained(self, z):
        return self.lower + self.width * np.exp(-_softplus(-z))

    def log_jacobian(self, z):
        # log |dx/dz| = log(width) + log(s) + log(1 - s), s = sigmoid(z)
        return np.log(self.width) - _softplus(-z) - _softplus(z)

    def random_init(self, rng):
        return rng.uniform(-2.0, 2.0)            # unconstrained space

    def __repr__(self):
        return f"Uniform({self.lower:g}, {self.upper:g})"


class Normal:
    """
    Normal(mu, sigma) prior.

        log p(x) = -0.5 log(2 pi sigma^2) - (x - mu)^2 / (2 sigma^2)

    Already unbounded, so it is sampled directly.
    """

    def __init__(self, mu, sigma):
        if not sigma > 0:
            raise ValueError("Normal prior needs sigma > 0.")
        self.mu, self.sigma = float(mu), float(sigma)

    def logpdf(self, x):
        return (-0.5 * np.log(2 * np.pi * self.sigma ** 2)
                - (x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def from_unconstrained(self, z):
        return z

    def log_jacobian(self, z):
        return 0.0

    def random_init(self, rng):
        return self.mu + self.sigma * rng.uniform(-1.0, 1.0)

    def __repr__(self):
        return f"Normal({self.mu:g}, {self.sigma:g})"


def log_prior(theta, priors):
    """Sum of the log prior densities of each parameter."""
    return float(sum(p.logpdf(t) for p, t in zip(priors, theta)))


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
# Warmup schedule (Stan's windowed adaptation)
# ---------------------------------------------------------------------------
def _adaptation_windows(n_warmup):
    """
    Slow-adaptation windows (start, end) for the mass matrix.

    An initial fast buffer tunes only the step size, then windows of
    doubling length (25, 50, 100, ...) each estimate a new mass matrix,
    and a final buffer re-tunes the step size for the last mass matrix.
    """
    if n_warmup < 20:
        return []
    init, term, base = 75, 50, 25
    if init + term + base > n_warmup:
        init = int(0.15 * n_warmup)
        term = int(0.10 * n_warmup)
        base = n_warmup - init - term
    windows = []
    start, size = init, base
    end_slow = n_warmup - term
    while start < end_slow:
        end = start + size
        if end + 2 * size > end_slow:    # next window would not fit: stretch
            end = end_slow
        windows.append((start, end))
        start, size = end, size * 2
    return windows


# ---------------------------------------------------------------------------
# Core NUTS sampler (works on any unconstrained log density)
# ---------------------------------------------------------------------------
def nuts(logp, x0, n_samples=2000, n_warmup=1000, grad=None,
         target_accept=0.8, max_depth=10, metric="diag", rng=None):
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
        Raise it (e.g. 0.95) if you see divergences.
    max_depth : int
        Maximum tree depth (at most 2**max_depth leapfrog steps per draw).
    metric : {"diag", "dense"}
        Mass matrix adapted during warmup. "dense" also learns the
        correlations between parameters, which helps for narrow,
        tilted posteriors (strongly correlated parameters).
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict with keys
        "samples"      (n_samples, d) draws in the unconstrained space
        "step_size"    final step size
        "inv_metric"   final inverse mass matrix, shape (d, d)
        "accept_rate"  mean acceptance statistic after warmup
        "n_divergent"  number of divergent transitions after warmup
        "divergent"    (n_samples,) bool, True where a draw diverged
        "tree_depth"   (n_samples,) tree depth of each draw
        "max_depth"    the max_depth setting
        "n_grad"       number of gradient evaluations after warmup
    """
    if metric not in ("diag", "dense"):
        raise ValueError("metric must be 'diag' or 'dense'.")
    rng = np.random.default_rng() if rng is None else rng
    grad = (lambda z: numerical_grad(logp, z)) if grad is None else grad
    x = np.asarray(x0, dtype=float).copy()
    d = x.size
    DELTA_MAX = 1000.0               # energy error that counts as divergence

    # Metric: kinetic energy K(r) = 0.5 r^T Minv r, momentum r ~ N(0, M)
    met = {"Minv": np.eye(d), "L": np.eye(d)}   # L = chol(M)

    def set_metric(Minv):
        met["Minv"] = Minv
        met["L"] = np.linalg.cholesky(np.linalg.inv(Minv))

    counters = {"n_grad": 0}

    def logp_and_grad(z):
        counters["n_grad"] += 1
        lp = logp(z)
        if lp is None or not np.isfinite(lp):
            return -np.inf, np.zeros(d)
        g = grad(z)
        if not np.all(np.isfinite(g)):
            return -np.inf, np.zeros(d)
        return float(lp), g

    def kinetic(r):
        return 0.5 * r @ met["Minv"] @ r

    def draw_momentum():
        return met["L"] @ rng.standard_normal(d)

    def leapfrog(z, r, g, eps):
        r = r + 0.5 * eps * g
        z = z + eps * (met["Minv"] @ r)
        lp, g = logp_and_grad(z)
        r = r + 0.5 * eps * g
        return z, r, g, lp

    def no_uturn(z_minus, z_plus, r_minus, r_plus):
        dz = z_plus - z_minus
        return (dz @ (met["Minv"] @ r_minus) >= 0 and
                dz @ (met["Minv"] @ r_plus) >= 0)

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

    windows = _adaptation_windows(n_warmup)
    window_ends = {end - 1: start for start, end in windows}
    window_draws = []

    samples = np.empty((n_samples, d))
    depths = np.empty(n_samples, dtype=int)
    divergent = np.zeros(n_samples, dtype=bool)
    accept_stats = np.empty(n_samples)

    for m in range(n_warmup + n_samples):
        if m == n_warmup:
            counters["n_grad"] = 0
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

            if any(start <= m < end for start, end in windows):
                window_draws.append(x.copy())
            if m in window_ends:
                w_draws = np.array(window_draws)
                k = len(w_draws)
                shrink = 1e-3 * (5.0 / (k + 5.0))
                if metric == "dense":
                    cov = np.atleast_2d(np.cov(w_draws, rowvar=False))
                    Minv = (k / (k + 5.0)) * cov + shrink * np.eye(d)
                else:
                    var = w_draws.var(axis=0, ddof=1)
                    Minv = np.diag((k / (k + 5.0)) * var + shrink)
                set_metric(Minv)
                window_draws = []
                eps = find_reasonable_eps(x, lp, g)
                da = init_dual_averaging(eps)

            if m == n_warmup - 1:
                eps = np.exp(da["log_eps_bar"])   # freeze step size
        else:
            i = m - n_warmup
            samples[i] = x
            depths[i] = j
            divergent[i] = state["divergent"]
            accept_stats[i] = accept

    return {
        "samples": samples,
        "step_size": eps,
        "inv_metric": met["Minv"],
        "accept_rate": float(accept_stats.mean()) if n_samples else np.nan,
        "n_divergent": int(divergent.sum()),
        "divergent": divergent,
        "tree_depth": depths,
        "max_depth": max_depth,
        "n_grad": counters["n_grad"],
    }


# ---------------------------------------------------------------------------
# Posterior sampling with priors
# ---------------------------------------------------------------------------
def sample(log_likelihood, priors, n_chains=4, n_samples=2000, n_warmup=1000,
           target_accept=0.8, max_depth=10, metric="diag", seed=None,
           args=(), verbose=True):
    """
    Sample a posterior with NUTS: log posterior = log likelihood + log prior.

    Parameters
    ----------
    log_likelihood : callable
        log_likelihood(theta, *args) -> float. Return ONLY the
        log-likelihood; the log prior is added here from `priors`.
    priors : list of Uniform / Normal
        One prior per parameter, in the same order as theta.
    n_chains, n_samples, n_warmup : int
    target_accept : float
    max_depth : int
    metric : {"diag", "dense"}
    seed : int, optional
    args : tuple, optional
        Extra arguments passed on to log_likelihood (e.g. data, inv_cov).
    verbose : bool
        Print one line of sampler statistics per chain.

    Returns
    -------
    chains : ndarray, shape (n_chains, n_samples, n_params)
        Posterior draws in the original parameter units.
    info : list of dict
        Per-chain sampler statistics (see `nuts`).
    """
    priors = list(priors)
    d = len(priors)
    rng = np.random.default_rng(seed)

    def to_theta(z):
        return np.array([p.from_unconstrained(zi) for p, zi in zip(priors, z)])

    def log_post(z):
        theta = to_theta(z)
        lp = log_prior(theta, priors)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta, *args)
        if ll is None or not np.isfinite(ll):
            return -np.inf
        log_jac = sum(p.log_jacobian(zi) for p, zi in zip(priors, z))
        return ll + lp + log_jac

    chains = np.empty((n_chains, n_samples, d))
    info = []
    for c in range(n_chains):
        z0 = np.array([p.random_init(rng) for p in priors])
        for _ in range(100):                     # find a finite start
            if np.isfinite(log_post(z0)):
                break
            z0 = np.array([p.random_init(rng) for p in priors])
        res = nuts(log_post, z0, n_samples=n_samples, n_warmup=n_warmup,
                   target_accept=target_accept, max_depth=max_depth,
                   metric=metric, rng=rng)
        chains[c] = np.array([to_theta(z) for z in res["samples"]])
        info.append(res)
        if verbose:
            n_max = int((res["tree_depth"] >= max_depth).sum())
            print(f"chain {c + 1}/{n_chains}: step size {res['step_size']:.3f}, "
                  f"accept {res['accept_rate']:.2f}, "
                  f"divergences {res['n_divergent']}, "
                  f"mean tree depth {res['tree_depth'].mean():.1f}"
                  + (f", hit max depth {n_max}x" if n_max else ""))
    return chains, info


def sample_posterior(log_likelihood, a, b, mu, sigma, **kwargs):
    """Two parameters with priors theta1 ~ Uniform(a, b), theta2 ~ Normal(mu, sigma)."""
    return sample(log_likelihood, [Uniform(a, b), Normal(mu, sigma)], **kwargs)


# ---------------------------------------------------------------------------
# Diagnostics: R-hat and effective sample size (Vehtari et al. 2021)
# ---------------------------------------------------------------------------
def _split(x):
    """Split each chain in half: (m, n) -> (2m, n // 2)."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)


def _rank_normalize(x):
    """Replace draws by normal scores of their pooled ranks."""
    from scipy.special import ndtri
    from scipy.stats import rankdata
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def _rhat_basic(x):
    """Classic (Gelman-Rubin) R-hat for chains x of shape (m, n)."""
    n = x.shape[1]
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    if W == 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def _ess_basic(x):
    """ESS of chains x of shape (m, n), Geyer's initial monotone sequence."""
    m, n = x.shape
    if n < 4 or np.all(x == x.flat[0]):
        return np.nan
    xc = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n   # biased autocov
    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = chain_var.mean()
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += x.mean(axis=1).var(ddof=1)
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0

    # Sum autocorrelation pairs while they stay positive, keeping them monotone
    tau = 0.0
    prev_pair = np.inf
    t = 0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev_pair)
        tau += pair
        prev_pair = pair
        t += 2
    tau = max(-1.0 + 2.0 * tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def rhat(x):
    """
    Rank-normalised split R-hat for one parameter, x shape (chains, draws).

    Compares between-chain and within-chain spread, also on the "folded"
    draws |x - median| so differences in scale are caught too.
    Converged chains give values close to 1; aim for R-hat < 1.01.
    """
    s = _split(np.asarray(x, dtype=float))
    r_bulk = _rhat_basic(_rank_normalize(s))
    folded = np.abs(s - np.median(s))
    r_tail = _rhat_basic(_rank_normalize(folded))
    return max(r_bulk, r_tail)


def ess_bulk(x):
    """Bulk effective sample size: how many independent draws the chains
    are worth for estimating the centre (mean, median) of the posterior."""
    return _ess_basic(_rank_normalize(_split(np.asarray(x, dtype=float))))


def ess_tail(x):
    """Tail effective sample size: the same for the 5% and 95% quantiles,
    i.e. for credible intervals."""
    s = _split(np.asarray(x, dtype=float))
    q05, q95 = np.quantile(s, [0.05, 0.95])
    return min(_ess_basic((s <= q05).astype(float)),
               _ess_basic((s <= q95).astype(float)))


def summary(chains, names=None, digits=4, print_table=True):
    """
    Posterior summary per parameter: mean, sd, 2.5/50/97.5 percentiles,
    R-hat, bulk ESS and tail ESS.

    Returns a list of dicts (one per parameter) and prints a table.
    """
    chains = np.asarray(chains)
    n_par = chains.shape[2]
    names = names or [f"theta{i + 1}" for i in range(n_par)]
    rows = []
    for i, name in enumerate(names):
        x = chains[:, :, i]
        flat = x.ravel()
        q = np.percentile(flat, [2.5, 50, 97.5])
        rows.append({"param": name, "mean": flat.mean(), "sd": flat.std(ddof=1),
                     "q2.5": q[0], "q50": q[1], "q97.5": q[2],
                     "r_hat": rhat(x), "ess_bulk": ess_bulk(x),
                     "ess_tail": ess_tail(x)})
    if print_table:
        f = f"{{:>{digits + 6}.{digits}f}}"
        print(f"{'param':>8} {'mean':>{digits + 6}} {'sd':>{digits + 6}} "
              f"{'2.5%':>{digits + 6}} {'50%':>{digits + 6}} {'97.5%':>{digits + 6}} "
              f"{'r_hat':>6} {'ess_bulk':>9} {'ess_tail':>9}")
        for r in rows:
            print(f"{r['param']:>8} " + " ".join(f.format(r[k]) for k in
                  ("mean", "sd", "q2.5", "q50", "q97.5"))
                  + f" {r['r_hat']:6.3f} {r['ess_bulk']:9.0f} {r['ess_tail']:9.0f}")
    return rows


def check_convergence(chains, info, names=None, rhat_max=1.01,
                      ess_min_per_chain=100, verbose=True):
    """
    Apply the standard convergence checks and say which ones fail.

    Checks (Vehtari et al. 2021; Stan's defaults):
      * R-hat < 1.01 for every parameter: all chains agree.
      * bulk and tail ESS > 100 per chain (400 for 4 chains): enough
        effectively independent draws for means and intervals.
      * no divergent transitions after warmup: no region of the posterior
        the sampler could not explore (a sign of biased results).
      * few draws hitting the maximum tree depth: the sampler is not
        being cut short (an efficiency warning rather than a bias one).

    Returns True if every check passes.
    """
    chains = np.asarray(chains)
    n_chains, n_draws, n_par = chains.shape
    names = names or [f"theta{i + 1}" for i in range(n_par)]
    ess_min = ess_min_per_chain * n_chains
    problems = []

    for i, name in enumerate(names):
        x = chains[:, :, i]
        r, eb, et = rhat(x), ess_bulk(x), ess_tail(x)
        if not r < rhat_max:
            problems.append(f"{name}: R-hat = {r:.3f} (should be < {rhat_max})")
        if not eb > ess_min:
            problems.append(f"{name}: bulk ESS = {eb:.0f} (should be > {ess_min})")
        if not et > ess_min:
            problems.append(f"{name}: tail ESS = {et:.0f} (should be > {ess_min})")

    n_div = sum(res["n_divergent"] for res in info)
    if n_div:
        problems.append(f"{n_div} divergent transitions after warmup "
                        f"(raise target_accept, or reparameterise)")
    n_max = sum(int((res["tree_depth"] >= res["max_depth"]).sum()) for res in info)
    if n_max > 0.01 * n_chains * n_draws:
        problems.append(f"{n_max} draws hit the maximum tree depth "
                        f"(sampler inefficient; try metric='dense')")

    if verbose:
        print(f"divergent transitions: {n_div}   "
              f"draws at max tree depth: {n_max}")
        if problems:
            print("CONVERGENCE CHECK FAILED:")
            for p in problems:
                print("  -", p)
        else:
            print(f"Convergence check passed: R-hat < {rhat_max}, "
                  f"ESS > {ess_min}, no divergences.")
    return not problems


# ---------------------------------------------------------------------------
# Plot styling shared by the plots below
# ---------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


# ---------------------------------------------------------------------------
# Trace plot
# ---------------------------------------------------------------------------
def trace_plot(chains, info=None, names=None, filename=None):
    """
    Trace plot: for each parameter, the per-chain marginal (left) and the
    draws against iteration (right). Divergent draws are marked with red
    ticks under the traces when `info` is given.

    Converged chains overlap: the left curves agree and the right traces
    look like the same stationary noise, with no trends or stuck stretches.
    """
    import matplotlib.pyplot as plt

    chains = np.asarray(chains)
    n_chains, n_draws, n_par = chains.shape
    names = names or [rf"$\theta_{i + 1}$" for i in range(n_par)]

    fig, axes = plt.subplots(n_par, 2, figsize=(11, 2.3 * n_par),
                             gridspec_kw={"width_ratios": [1, 3]}, squeeze=False)
    for i in range(n_par):
        ax_h, ax_t = axes[i]
        lo, hi = np.percentile(chains[:, :, i], [0.1, 99.9])
        for c in range(n_chains):
            col = SERIES[c % len(SERIES)]
            ax_h.hist(chains[c, :, i], bins=40, range=(lo, hi), histtype="step",
                      density=True, color=col, linewidth=1.3)
            ax_t.plot(chains[c, :, i], color=col, linewidth=0.5, alpha=0.75,
                      label=f"chain {c + 1}")
        if info is not None:
            div_idx = np.concatenate([np.flatnonzero(res["divergent"]) for res in info])
            if div_idx.size:
                ymin = ax_t.get_ylim()[0]
                ax_t.plot(div_idx, np.full(div_idx.size, ymin), "|", color="#e34948",
                          markersize=10, label="divergent")
        for ax in (ax_h, ax_t):
            _style_axis(ax)
        ax_h.set_yticks([])
        ax_h.spines["left"].set_visible(False)
        ax_h.set_xlabel(names[i], color=INK, fontsize=10)
        ax_t.set_ylabel(names[i], color=INK, fontsize=10)
        if i < n_par - 1:
            ax_t.set_xticklabels([])
    axes[-1, 1].set_xlabel("draw (after warmup)", color=INK)
    axes[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=8,
                      ncol=n_chains + 1, frameon=False)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------
def credible_thresholds(H, levels):
    """Density thresholds enclosing the given probability masses."""
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    idx = [min(np.searchsorted(cum, lv), flat.size - 1) for lv in levels]
    return sorted(set(flat[i] for i in idx))


def joint_density(x, y, bins=40, smooth=1.0, ranges=None):
    """Smoothed 2D histogram of samples: returns (x_centres, y_centres, H)."""
    if ranges is None:
        ranges = [np.percentile(x, [0.1, 99.9]), np.percentile(y, [0.1, 99.9])]
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=ranges)
    if smooth:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, smooth)
        except ImportError:
            pass
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H


def corner_plot(chains, names=None, truths=None, bins=40, smooth=1.0,
                levels=(0.393, 0.865), color="#2a78d6", filename=None):
    """
    Corner plot of posterior samples: 1D marginals on the diagonal and
    2D joint densities (filled credible regions) below it.

    Parameters
    ----------
    chains : ndarray, shape (n_chains, n_samples, n_params) or (n_draws, n_params)
        Posterior draws, e.g. the output of `sample`.
    names : list of str, optional
        Parameter labels (LaTeX allowed, e.g. r"$H_0$").
    truths : list of float, optional
        Reference values to mark (e.g. true values in a simulation).
    bins : int
        Number of histogram bins per axis.
    smooth : float
        Gaussian smoothing (in bins) of the 2D histograms; 0 for none.
    levels : tuple of float
        Probability mass inside each 2D contour. The defaults 0.393 and
        0.865 are the 2D equivalents of 1-sigma and 2-sigma.
    color : str
        Colour of the marks.
    filename : str, optional
        If given, the figure is saved there (e.g. "corner.png").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    samples = np.asarray(chains)
    if samples.ndim == 3:
        samples = samples.reshape(-1, samples.shape[-1])
    n_par = samples.shape[1]
    names = names or [rf"$\theta_{i + 1}$" for i in range(n_par)]

    ranges = []
    for i in range(n_par):
        lo, hi = np.percentile(samples[:, i], [0.1, 99.9])
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    size = 2.6 * n_par
    fig, axes = plt.subplots(n_par, n_par, figsize=(size, size), squeeze=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)

    for i in range(n_par):
        for j in range(n_par):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            _style_axis(ax)

            if i == j:
                # 1D marginal with median and 68% interval
                x = samples[:, i]
                ax.hist(x, bins=bins, range=ranges[i], histtype="stepfilled",
                        color=to_rgba(color, 0.18), edgecolor=color, linewidth=1.5)
                q16, q50, q84 = np.percentile(x, [16, 50, 84])
                for q, ls in ((q16, "--"), (q50, "-"), (q84, "--")):
                    ax.axvline(q, color=INK, linestyle=ls, linewidth=1)
                ax.set_title(f"{names[i]} = {q50:.3f}"
                             f"$^{{+{q84 - q50:.3f}}}_{{-{q50 - q16:.3f}}}$",
                             fontsize=10, color=INK)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.set_xlim(ranges[i])
            else:
                # 2D joint density with credible-region contours
                x, y = samples[:, j], samples[:, i]
                xc, yc, H = joint_density(x, y, bins=bins, smooth=smooth,
                                          ranges=[ranges[j], ranges[i]])
                thresholds = credible_thresholds(H, levels)
                step = max(1, len(x) // 3000)
                ax.plot(x[::step], y[::step], ".", color=MUTED, alpha=0.15,
                        markersize=2, zorder=0)
                fill_levels = list(thresholds) + [H.max() * 1.001]
                alphas = np.linspace(0.3, 0.75, len(thresholds))
                ax.contourf(xc, yc, H.T, levels=fill_levels,
                            colors=[to_rgba(color, a) for a in alphas], zorder=1)
                ax.contour(xc, yc, H.T, levels=thresholds, colors=[color],
                           linewidths=1.2, zorder=2)
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])

            if truths is not None:
                if truths[j] is not None:
                    ax.axvline(truths[j], color="#eb6834", linewidth=1.2)
                if i != j and truths[i] is not None:
                    ax.axhline(truths[i], color="#eb6834", linewidth=1.2)
                    ax.plot(truths[j], truths[i], "s", color="#eb6834", markersize=5)

            # Labels only on the outer edge
            if i == n_par - 1:
                ax.set_xlabel(names[j], color=INK, fontsize=11)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], color=INK, fontsize=11)
            elif j > 0:
                ax.set_yticklabels([])

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Example: straight-line fit y = theta1 * x + theta2 + noise
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_data = np.linspace(0, 5, 40)
    y_data = 3.0 * x_data + 4.0 + rng.normal(0, 1.0, x_data.size)

    def log_likelihood(theta, x, y, noise):
        slope, intercept = theta
        resid = y - (slope * x + intercept)
        return (-0.5 * np.sum(resid ** 2) / noise ** 2
                - x.size * np.log(noise * np.sqrt(2 * np.pi)))

    priors = [Uniform(0.0, 10.0), Normal(5.0, 2.0)]
    chains, info = sample(log_likelihood, priors, seed=1,
                          args=(x_data, y_data, 1.0))
    print()
    summary(chains, names=["slope", "intercept"])
    check_convergence(chains, info, names=["slope", "intercept"])
    trace_plot(chains, info, names=["slope", "intercept"], filename="trace_example.png")
    corner_plot(chains, names=["slope", "intercept"], truths=[3.0, 4.0],
                filename="corner_example.png")
