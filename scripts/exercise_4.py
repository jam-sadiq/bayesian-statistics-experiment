"""
Exercise #4: H0 and MB from Pantheon+ SH0ES supernovae with NUTS.

Runs the inference for several priors on MB and two data selections:

  * "HF only"   - Hubble-flow SNe only (USED_IN_SH0ES_HF == 1). These
                  constrain only M = MB - 5 log10(H0 / H0ref).
  * "HF + cal"  - Hubble flow plus the Cepheid-calibrated SNe
                  (IS_CALIBRATOR == 1), which break the H0-MB degeneracy.

For each case it prints the posterior summary with R-hat, ESS and
divergences, and saves a trace plot and a corner plot of (H0, MB, M).
At the end it saves comparison plots across the cases.
Needs nuts.py in the same folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

from nuts import (sample, Uniform, Normal, log_prior, summary,
                  check_convergence, trace_plot, corner_plot,
                  joint_density, credible_thresholds, SERIES, INK, MUTED, GRID)

# ---------------------------------------------------------------
# 1. Read the .dat file, extract the 5 columns, build the masks
# ---------------------------------------------------------------
datfile = '../data/Pantheon+SH0ES.dat'
covfile = '../data/Pantheon+SH0ES_STAT+SYS.cov'

df = pd.read_csv(datfile, sep=r'\s+')      # first line = column names

zHDvals        = df['zHD'].values
m_b_corr_vals  = df['m_b_corr'].values
CEPH_DIST_vals = df['CEPH_DIST'].values
flag_calib     = df['IS_CALIBRATOR'].values
flag_hf        = df['USED_IN_SH0ES_HF'].values

mask_full = (flag_calib == 1) | (flag_hf == 1)   # calibrators + Hubble flow
mask_hf   = (flag_hf == 1) & (flag_calib == 0)   # Hubble flow only

# ---------------------------------------------------------------
# 2. Covariance: skip first line (N), reshape to N x N
# ---------------------------------------------------------------
cov_vals = np.loadtxt(covfile)              # first value is N
N = int(cov_vals[0])
C = cov_vals[1:].reshape(N, N)
assert N == len(df), "N in .cov file does not match rows in .dat file"

# ---------------------------------------------------------------
# 3. Luminosity distance and distance modulus
# ---------------------------------------------------------------
c = 299792.458  # km/s
H0ref = 70.0

def luminosity_distance(z, H0, Om=0.3):
    """Luminosity distance in Mpc for flat LCDM (radiation neglected)."""
    OL = 1.0 - Om
    E = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + OL)
    integral, _ = quad(E, 0, z)
    return (1 + z) * (c / H0) * integral


def build_data(mask):
    """Everything the likelihood needs for one selection of supernovae."""
    z = zHDvals[mask]
    C_masked = C[np.ix_(mask, mask)]        # same rows AND columns
    # dL is proportional to 1/H0, so do the integrals ONCE (with H0=1)
    dL_times_H0 = np.array([luminosity_distance(zi, 1.0) for zi in z])
    return {
        "m_obs": m_b_corr_vals[mask],
        "ceph": CEPH_DIST_vals[mask],
        "is_cal": flag_calib[mask] == 1,
        "dL_times_H0": dL_times_H0,
        "inv_cov": np.linalg.inv(C_masked),
    }


def mu1_model(H0, dL_times_H0):
    """mu from redshift: 5 log10(dL / Mpc) + 25, with dL = dL_times_H0 / H0."""
    mu_ref = 5 * np.log10(dL_times_H0 / H0ref) + 25     # mu for H0 = H0ref
    return mu_ref - 5 * np.log10(H0 / H0ref)

# ---------------------------------------------------------------
# 4. Residual and log-likelihood
# ---------------------------------------------------------------
def residual(Mb, H0, data):
    # Calibrators: distance from Cepheids (mu2 = CEPH_DIST)
    # Hubble flow: distance from redshift (mu1 = 5 log10 dL + 25)
    mu = np.where(data["is_cal"], data["ceph"], mu1_model(H0, data["dL_times_H0"]))
    m_model = Mb + mu
    return data["m_obs"] - m_model


def log_likelihood(theta, data):
    """Gaussian log-likelihood with the full covariance. theta = [H0, Mb]."""
    H0, Mb = theta
    r = residual(Mb, H0, data)
    return -0.5 * r @ data["inv_cov"] @ r


def M_combination(H0, Mb):
    """The combination the Hubble flow constrains: M = MB - 5 log10(H0/H0ref)."""
    return Mb - 5 * np.log10(H0 / H0ref)

# ---------------------------------------------------------------
# 5. Priors, written out explicitly
# ---------------------------------------------------------------
def log_uniform_pdf(x, a, b):
    if a <= x <= b:
        return -np.log(b - a)
    return -np.inf

def log_gaussian_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)

H0_bounds = (50, 100)

def log_prior_previous(theta):
    """Previous run: H0 ~ U(50, 100), MB ~ N(-19.3, 0.01)."""
    H0, Mb = theta
    return log_uniform_pdf(H0, *H0_bounds) + log_gaussian_pdf(Mb, -19.3, 0.01)

def log_prior_gauss(theta):
    """H0 ~ U(50, 100), MB ~ N(-19.2, 0.12)."""
    H0, Mb = theta
    return log_uniform_pdf(H0, *H0_bounds) + log_gaussian_pdf(Mb, -19.2, 0.12)

def log_prior_flat(theta):
    """H0 ~ U(50, 100), MB ~ U(-20.5, -18)."""
    H0, Mb = theta
    return log_uniform_pdf(H0, *H0_bounds) + log_uniform_pdf(Mb, -20.5, -18.0)

# The same priors as objects for the sampler (it needs them to know which
# parameters are bounded). The check below confirms they are identical.
PRIORS = {
    "tight": ([Uniform(*H0_bounds), Normal(-19.3, 0.01)], log_prior_previous,
              r"$M_B\sim\mathcal{N}(-19.3,\ 0.01)$"),
    "gauss": ([Uniform(*H0_bounds), Normal(-19.2, 0.12)], log_prior_gauss,
              r"$M_B\sim\mathcal{N}(-19.2,\ 0.12)$"),
    "flat":  ([Uniform(*H0_bounds), Uniform(-20.5, -18.0)], log_prior_flat,
              r"$M_B\sim\mathcal{U}(-20.5,\ -18)$"),
}
_rng = np.random.default_rng(0)
for key, (priors, fn, _) in PRIORS.items():
    for _ in range(100):
        th = [_rng.uniform(40, 110), _rng.uniform(-21, -17.5)]
        a_, b_ = fn(th), log_prior(th, priors)
        assert (a_ == b_ == -np.inf) or np.isclose(a_, b_), key

# ---------------------------------------------------------------
# 6. Run the cases
# ---------------------------------------------------------------
DATA = {"hf": build_data(mask_hf), "full": build_data(mask_full)}
DATA_LABEL = {"hf": "HF only", "full": "HF + calibrators"}
print(f"Total SNe: {len(df)}; HF only: {mask_hf.sum()}; "
      f"HF + calibrators: {mask_full.sum()} ({(flag_calib == 1).sum()} calibrators)")

# (data, prior) pairs; comment out any you don't need
CASES = [
    ("full", "tight"),   # the previous run
    ("hf",   "tight"),   # Q7: tight prior, Hubble flow only
    ("hf",   "gauss"),   # Q4
    ("hf",   "flat"),    # Q5
    ("full", "gauss"),   # Q8: calibrators added
    ("full", "flat"),    # Q8
]

names3 = [r"$H_0$", r"$M_B$", r"$\mathcal{M}$"]
results = {}

for data_key, prior_key in CASES:
    case = f"{data_key}_{prior_key}"
    priors, _, prior_label = PRIORS[prior_key]
    print("\n" + "=" * 70)
    print(f"{case}:  {DATA_LABEL[data_key]},  H0 ~ {priors[0]},  MB ~ {priors[1]}")
    print("=" * 70)

    chains, info = sample(log_likelihood, priors, n_chains=4, n_samples=2000,
                          n_warmup=1000, seed=42, args=(DATA[data_key],))

    # Derived quantity for every posterior sample
    M = M_combination(chains[..., 0], chains[..., 1])
    chains3 = np.concatenate([chains, M[..., None]], axis=-1)

    summary(chains3, names=["H0", "MB", "M"])
    ok = check_convergence(chains, info, names=["H0", "MB"])

    trace_plot(chains3, info, names=names3, filename=f"trace_{case}.png")
    corner_plot(chains3, names=names3, filename=f"corner_{case}.png")
    plt.close("all")
    np.savez(f"chains_{case}.npz", H0=chains[..., 0], MB=chains[..., 1], M=M)

    results[case] = {"data": data_key, "prior": prior_key, "label": prior_label,
                     "H0": chains[..., 0].ravel(), "MB": chains[..., 1].ravel(),
                     "M": M.ravel(), "converged": ok}

# ---------------------------------------------------------------
# 7. Results table
# ---------------------------------------------------------------
rows = []
for case, r in results.items():
    row = {"case": case, "data": DATA_LABEL[r["data"]],
           "MB prior": str(PRIORS[r["prior"]][0][1])}
    for p in ("H0", "MB", "M"):
        row[f"{p} mean"] = r[p].mean()
        row[f"{p} sd"] = r[p].std(ddof=1)
    row["converged"] = r["converged"]
    rows.append(row)
table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n", table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
table.to_csv("results_table.csv", index=False)

# ---------------------------------------------------------------
# 8. Comparison plots
# ---------------------------------------------------------------
def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def joint_contours(ax, H0, MB, color, label, level=0.865, fill=True,
                   linestyle="-", dots=True):
    """Joint posterior of (H0, MB): faint posterior draws plus the
    2-sigma (2D) credible region enclosing `level` of the probability."""
    from matplotlib.colors import to_rgba
    if dots:
        step = max(1, len(H0) // 2500)
        ax.plot(H0[::step], MB[::step], ".", color=color, alpha=0.12,
                markersize=2.5, zorder=1)
    xc, yc, H = joint_density(H0, MB, bins=90, smooth=1.5,
                              ranges=[(48, 102), (-20.6, -17.9)])
    th = credible_thresholds(H, [level])
    if fill:
        ax.contourf(xc, yc, H.T, levels=list(th) + [H.max() * 1.001],
                    colors=[to_rgba(color, 0.2)], zorder=2)
    ax.contour(xc, yc, H.T, levels=th, colors=[color], linewidths=1.6,
               linestyles=linestyle, zorder=3)
    ax.plot([], [], color=color, linewidth=1.6, linestyle=linestyle, label=label)


H0_grid = np.linspace(50, 100, 200)

# (a) Joint (H0, MB) posterior, Hubble flow only, three priors
hf_cases = [c for c in ("hf_tight", "hf_gauss", "hf_flat") if c in results]
if hf_cases:
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, case in enumerate(hf_cases):
        r = results[case]
        joint_contours(ax, r["H0"], r["MB"], SERIES[k], r["label"])
    M_hat = np.median(results[hf_cases[-1]]["M"])
    ax.plot(H0_grid, M_hat + 5 * np.log10(H0_grid / H0ref), "--", color=INK,
            linewidth=1, label=r"$M_B = \hat{\mathcal{M}} + 5\log_{10}(H_0/70)$")
    ax.set_xlim(48, 102)
    ax.set_ylim(-20.6, -17.9)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", color=INK)
    ax.set_ylabel(r"$M_B$  [mag]", color=INK)
    ax.set_title("Joint posterior, Hubble-flow SNe only (86% regions)",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig("joint_hf_only.png", dpi=150)

    # (b) Marginals of H0 and of M for the same three priors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    prev = results.get("full_tight")
    if prev is not None:
        m, s = prev["H0"].mean(), prev["H0"].std()
        ax1.axvspan(m - s, m + s, color=GRID, alpha=0.9, lw=0,
                    label=f"previous run (HF + cal): {m:.2f} ± {s:.2f}")
    M_all = np.concatenate([results[c]["M"] for c in hf_cases])
    M_lo, M_hi = np.percentile(M_all, [0.05, 99.95])
    for k, case in enumerate(hf_cases):
        r = results[case]
        ax1.hist(r["H0"], bins=60, range=(50, 100), density=True, histtype="step",
                 color=SERIES[k], linewidth=1.6, label=r["label"])
        ax2.hist(r["M"], bins=50, range=(M_lo, M_hi), density=True, histtype="step",
                 color=SERIES[k], linewidth=1.6, label=r["label"])
    ax1.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", color=INK)
    ax2.set_xlabel(r"$\mathcal{M} = M_B - 5\log_{10}(H_0/70)$  [mag]", color=INK)
    ax1.set_title(r"$H_0$: changes with the $M_B$ prior", color=INK, fontsize=11)
    ax2.set_title(r"$\mathcal{M}$: the same for every prior", color=INK, fontsize=11)
    for ax in (ax1, ax2):
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        style(ax)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("marginals_H0_vs_M.png", dpi=150)

# (c) What the calibrators do: same flat prior, with and without them
cal_cases = [c for c in ("hf_flat", "full_flat", "full_gauss") if c in results]
if len(cal_cases) > 1:
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, case in enumerate(cal_cases):
        r = results[case]
        joint_contours(ax, r["H0"], r["MB"], SERIES[k],
                       f"{DATA_LABEL[r['data']]}, {r['label']}",
                       fill=(k == 0), linestyle=("-", "-", "--")[k], dots=(k == 0))
    ax.set_xlim(48, 102)
    ax.set_ylim(-20.6, -17.9)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", color=INK)
    ax.set_ylabel(r"$M_B$  [mag]", color=INK)
    ax.set_title("Adding Cepheid-calibrated SNe breaks the degeneracy (86% regions)",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig("joint_with_calibrators.png", dpi=150)

print("\nSaved: trace_*.png, corner_*.png, chains_*.npz, results_table.csv,")
print("       joint_hf_only.png, marginals_H0_vs_M.png, joint_with_calibrators.png")
plt.show()
