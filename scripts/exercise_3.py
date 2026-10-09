"""
Exercise #3:
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
    """mu from redshift: 5 log10(dL) + 25.
    change from 174 in notes"""
    DL = dL_times_H0# / H0
    H0ref = 70
    mu_ref = 5 * np.log10(DL/H0ref) + 25
    return mu_ref - 5* np.log10(H0/H0ref)

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
H0_bounds = (50, 100)          # Uniform prior on H0: (lower, upper)
Mb_bounds = (-19.3, 0.01)      # Gaussian prior on Mb: (mean, sigma)

def log_unformprior(theta):
    """Uniform prior: log(1)=0 inside the box, log(0)=-inf outside."""
    H0, Mb = theta
    if H0_bounds[0] < H0 < H0_bounds[1] and Mb_bounds[0] < Mb < Mb_bounds[1]:
        return 0.0
    return -np.inf


def log_uniform_pdf(x, a, b):
    if a <= x <= b:
        return -np.log(b - a)
    return -np.inf

def log_gaussian_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)


def log_prior(theta, a=H0_bounds[0], b=H0_bounds[1], mu=Mb_bounds[0], sigma=Mb_bounds[1]):
    """
    Log of the joint density of two independent parameters.

    theta1 follows a Uniform(a, b) distribution and theta2 follows a
    Normal(mu, sigma) distribution. Because they are independent, the
    joint density is the product of the two densities, so its log is
    the sum of the two log densities:

        log p(theta1, theta2) = log p_U(theta1) + log p_N(theta2)

    Parameters
    ----------
    theta : array_like of length 2
        The two parameter values ``(theta1, theta2)``.
    a : float
        Lower bound of the uniform distribution for theta1.
    b : float
        Upper bound of the uniform distribution for theta1. Must be > a.
    mu : float
        Mean of the Gaussian distribution for theta2.
    sigma : float
        Standard deviation of the Gaussian distribution for theta2.
        Must be > 0.

    Returns
    -------
    float
        The log joint density at ``theta``. Returns ``-np.inf`` if
        theta1 lies outside ``[a, b]``, where the density is zero.

    Examples
    --------
    >>> log_prior([3.0, 4.5], a=0.0, b=10.0, mu=5.0, sigma=2.0)
    -3.9395...
    """
    theta1, theta2 = theta
    lp = log_uniform_pdf(theta1, a, b)
    if not np.isfinite(lp):  # outside the uniform bounds
        return -np.inf
    return lp + log_gaussian_pdf(theta2, mu, sigma)

def log_posterior(theta, inv_cov, const=0.0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf             # outside bounds: skip the likelihood
    H0, Mb = theta
    return lp + log_likelihood(Mb, H0, inv_cov, const)


# ---------------------------------------------------------------
# 5. Sample the posterior with NUTS
# ---------------------------------------------------------------
from nuts import sample_posterior, summary, corner_plot, trace_plot

def log_likelihood_theta(theta, inv_cov):
    """Adapter for nuts.py, which passes theta = [theta1, theta2].

    theta1 = H0 (uniform prior), theta2 = Mb (gaussian prior).
    """
    H0, Mb = theta
    return log_likelihood(Mb, H0, inv_cov)

a, b = H0_bounds          # uniform prior on H0
mu, sigma = Mb_bounds     # gaussian prior on Mb

chains, info = sample_posterior(log_likelihood_theta, a, b, mu, sigma,
                                n_chains=4, n_samples=2000, n_warmup=1000,
                                seed=42, args=(inv_cov,))

names = [r"$H_0$", r"$M_b$"]
summary(chains, names=["H0", "Mb"])

# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------
trace_plot(chains, names=names, filename="trace.png")
corner_plot(chains, names=names, filename="corner.png")
print("Saved trace.png and corner.png")
plt.show()
