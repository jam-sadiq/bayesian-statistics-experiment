"""
Script from an exercise at the AIPHY2 Bayesian Statistics School,
GSSI, L'Aquila, 2026.

What I learned: how to use MCMC on a two-parameter problem with the
Metropolis-Hastings algorithm.

Data
----
Pantheon+SH0ES type Ia supernovae. For each supernova we have:
  - z          : redshift
  - m_b_corr   : corrected apparent magnitude (how bright it looks)
  - CEPH_DIST  : distance modulus mu from Cepheids (calibrators only)
plus a covariance matrix for the uncertainties on the magnitudes.

We keep only the supernovae used by SH0ES: the Cepheid calibrators and
the Hubble-flow sample.

Goal
----
Infer two parameters:
  - H0 : the Hubble constant [km/s/Mpc]
  - Mb : the absolute magnitude of a type Ia supernova (its intrinsic
         brightness, which is the same for every SN Ia after corrections)

Model
-----
Apparent and absolute magnitude are linked by the distance modulus:
    m_b = Mb + mu

The distance modulus mu comes from two sources:
  - calibrators : mu = CEPH_DIST (measured with Cepheids)
  - Hubble choice : mu = 5 log10(dL / Mpc) + 25, where the luminosity
                  distance dL depends on z and H0 (flat LCDM, Om = 0.3)

Residual and likelihood
-----------------------
residual = m_b_corr - (Mb + mu)

log L(H0, Mb) = -0.5 * residual^T  C^-1  residual

where C is the covariance matrix (masked to the same supernovae).

Prior
-----
Uniform prior: H0 in [60, 80], Mb in [-20, 20].
The prior density is constant inside this box and zero outside. Taking
the log, log prior = 0 inside and -infinity outside. So adding it to the
log-likelihood does nothing inside the bounds and rejects any point
outside them [ask this to instructor?]

Posterior
---------
log posterior = log likelihood + log prior   (up to a constant)

Metropolis-Hastings
-------------------
1. Start from an initial guess (H0, Mb) and compute its log posterior.
2. Propose a new point assuming a distrubution: Gaussian + (small)noise   (2D Gaussian centred on the CURRENT point, uncorrelated, small widths).
3. Compute the log posterior at the new point.
4. Acceptance probability: alpha = min(1, posterior_new / posterior_old). 
   In logs: log_alpha = min(0, logpost_new - logpost_old).
5. Draw u ~ Uniform(0, 1). If u < alpha, accept and move to the new
   point; otherwise stay at the current point.
6. Save the current point (repeated if rejected) and go back to step 2.

After many steps (5000 here), discard the first part of the chain
(burn-in), then make trace plots, a corner plot and the marginalized
posteriors of H0 and Mb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ---------------------------------------------------------------
# 1. Read the .dat file, extract the 5 columns, build the mask
# ---------------------------------------------------------------
datfile = '../data/Pantheon+SH0ES.dat'
covfile = '../data/Pantheon+SH0ES_STAT+SYS.cov'

df = pd.read_csv(datfile, sep=r'\s+')      # first line = column names

zHDvals        = df['zHD'].values
m_b_corr_vals  = df['m_b_corr'].values
CEPH_DIST_vals = df['CEPH_DIST'].values
flag_calib     = df['IS_CALIBRATOR'].values
flag_hf        = df['USED_IN_SH0ES_HF'].values

mask = (df['IS_CALIBRATOR'] == 1) | (df['USED_IN_SH0ES_HF'] == 1)
mask = mask.values

z      = zHDvals[mask]
m_obs  = m_b_corr_vals[mask]
ceph   = CEPH_DIST_vals[mask]
is_cal = flag_calib[mask] == 1

print(f"Total SNe: {len(df)},  after mask: {mask.sum()} "
      f"({is_cal.sum()} calibrators, {(~is_cal).sum()} Hubble flow)")

# ---------------------------------------------------------------
# 2. Covariance: skip first line (N), reshape to N x N, apply mask
# ---------------------------------------------------------------
cov_vals = np.loadtxt(covfile)              # first value is N
N = int(cov_vals[0])
C = cov_vals[1:].reshape(N, N)
assert N == len(df), "N in .cov file does not match rows in .dat file"

C_masked = C[np.ix_(mask, mask)]            # same rows AND columns
inv_cov  = np.linalg.inv(C_masked)
print("Masked covariance shape:", C_masked.shape)

# ---------------------------------------------------------------
# 3. Luminosity distance and distance modulus
# ---------------------------------------------------------------
c = 299792.458  # km/s

def luminosity_distance(z, H0, Om=0.3):
    """Luminosity distance in Mpc for flat LCDM (radiation neglected)."""
    OL = 1.0 - Om
    E = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + OL)
    integral, _ = quad(E, 0, z)
    return (1 + z) * (c / H0) * integral

# Speed-up: dL is proportional to 1/H0, so do the integrals ONCE (with H0=1)
# and just divide by H0 later. Same result, ~1000x faster inside the MCMC.
dL_times_H0 = np.array([luminosity_distance(zi, 1.0) for zi in z])

def mu1_model(H0):
    """mu from redshift: 5 log10(dL) + 25."""
    DL = dL_times_H0 / H0
    return 5 * np.log10(DL) + 25

# ---------------------------------------------------------------
# 4. Residual, log-likelihood, prior, posterior
# ---------------------------------------------------------------
def residual(Mb, H0):
    # Calibrators: distance from Cepheids (mu2 = CEPH_DIST)
    # Hubble flow: distance from redshift (mu1 = 5 log10 dL + 25)
    mu = np.where(is_cal, ceph, mu1_model(H0))
    m_model = Mb + mu
    return m_obs - m_model

def log_likelihood(Mb, H0, inv_cov, const=0.0):
    r = residual(Mb, H0)
    return -0.5 * r @ inv_cov @ r + const


#  check this with instructor
H0_bounds = (60, 80)
Mb_bounds = (-20, 20)

def log_prior(theta):
    """Uniform prior: log(1)=0 inside the box, log(0)=-inf outside."""
    H0, Mb = theta
    if H0_bounds[0] < H0 < H0_bounds[1] and Mb_bounds[0] < Mb < Mb_bounds[1]:
        return 0.0
    return -np.inf

def log_posterior(theta, inv_cov, const=0.0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf             # outside bounds: skip the likelihood
    H0, Mb = theta
    return lp + log_likelihood(Mb, H0, inv_cov, const)

# ---------------------------------------------------------------
# 5. Metropolis algorithm
# ---------------------------------------------------------------
def metropolis(theta0, inv_cov, n_steps=1000, step=(0.02, 0.001), seed=42):
    rng = np.random.default_rng(seed)
    prop_cov = np.diag(np.array(step)**2)    # uncorrelated Gaussian proposal

    theta = np.array(theta0, dtype=float)
    logp = log_posterior(theta, inv_cov)

    chain = [theta.copy()]
    n_accept = 0
    for i in range(n_steps):
        # new point = current point + Gaussian noise
        theta_new = rng.multivariate_normal(theta, prop_cov)
        logp_new = log_posterior(theta_new, inv_cov)

        # ratio = posterior_new / posterior_old  ->  in logs: difference
        log_alpha = min(0.0, logp_new - logp)
        u = rng.uniform(0, 1)
        if np.log(u) < log_alpha:            # accept
            theta, logp = theta_new, logp_new
            n_accept += 1
        chain.append(theta.copy())           # rejected -> repeat old values

    print(f"Acceptance rate: {n_accept / n_steps:.2f}")
    return np.array(chain)

chain = metropolis(theta0=[69.0, -19.0], inv_cov=inv_cov, n_steps=50000)

burn = len(chain) // 5
H0_s, Mb_s = chain[burn:, 0], chain[burn:, 1]
print(f"H0 = {H0_s.mean():.2f} +/- {H0_s.std():.2f}")
print(f"Mb = {Mb_s.mean():.3f} +/- {Mb_s.std():.3f}")

# ---------------------------------------------------------------
# 6. Trace plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(chain[:, 0], lw=0.8)
ax[0].set_ylabel('H0')
ax[1].plot(chain[:, 1], lw=0.8, color='C1')
ax[1].set_ylabel('Mb')
ax[1].set_xlabel('Step')
plt.tight_layout()
plt.savefig('small_jumps_trace_plots.png', dpi=120)
plt.show()

# ---------------------------------------------------------------
# 7. Corner plot (2D contours + marginalized 1D histograms)
#    pip install corner
# ---------------------------------------------------------------
import corner

samples = np.column_stack([H0_s, Mb_s])
fig = corner.corner(
    samples,
    labels=[r'$H_0$ [km/s/Mpc]', r'$M_b$'],
    quantiles=[0.16, 0.5, 0.84],         # dashed lines: median and 68% interval
    show_titles=True, title_fmt='.3f',   # value +/- error above each histogram
    levels=(0.68, 0.95),                 # 1-sigma and 2-sigma contours
    bins=30, smooth=1.0,
    fill_contours=True, plot_datapoints=False,
)
plt.savefig('small_jumps_corner_plot.png', dpi=120)
plt.show()

# ---------------------------------------------------------------
# 8. Marginalized posteriors, one per parameter
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, s, name in zip(ax, [H0_s, Mb_s], [r'$H_0$ [km/s/Mpc]', r'$M_b$']):
    lo, med, hi = np.percentile(s, [16, 50, 84])
    a.hist(s, bins=40, density=True, histtype='stepfilled', alpha=0.6)
    a.axvline(med, color='k')
    a.axvline(lo, color='k', ls='--')
    a.axvline(hi, color='k', ls='--')
    a.set_xlabel(name)
    a.set_ylabel('Posterior density')
    a.set_title(f'{med:.3f}  (+{hi - med:.3f} / -{med - lo:.3f})')
plt.tight_layout()
plt.savefig('small_jumps_marginalized_plots.png', dpi=120)
plt.show()



# Per-supernova estimates from the .dat data
Mb_data = m_obs[is_cal] - ceph[is_cal]                       # calibrators: Mb = m_b - mu_Cepheid
mu_hf   = m_obs[~is_cal] - np.median(Mb_s)                    # Hubble flow: mu = m_b - Mb
H0_data = dL_times_H0[~is_cal] / 10**((mu_hf - 25) / 5)       # H0 = (dL * H0) / dL
# check
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(H0_data, bins=30, density=True, alpha=0.4, label='Data (per SN)')
ax[0].hist(H0_s,    bins=40, density=True, alpha=0.7, label='Posterior')
ax[0].set_xlabel(r'$H_0$ [km/s/Mpc]'); ax[0].legend()
ax[1].hist(Mb_data, bins=20, density=True, alpha=0.4, label='Data (per SN)')
ax[1].hist(Mb_s,    bins=40, density=True, alpha=0.7, label='Posterior')
ax[1].set_xlabel(r'$M_b$'); ax[1].legend()
plt.tight_layout(); plt.savefig('small_jumps_data_vs_posterior.png', dpi=120); plt.show()
