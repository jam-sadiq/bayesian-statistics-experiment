import numpy as np
import pandas as pd
from scipy.integrate import quad

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Read data and apply the same mask as before
# ---------------------------------------------------------------
datfile = '../data/Pantheon+SH0ES.dat'
df = pd.read_csv(datfile, sep=r'\s+')

mask = ((df['IS_CALIBRATOR'] == 1) | (df['USED_IN_SH0ES_HF'] == 1)).values
dfm = df[mask].reset_index(drop=True)        # 354 rows; row i here = masked index i
orig_idx = np.where(mask)[0]                 # row number of each one in the full .dat

z      = dfm['zHD'].values
m_obs  = dfm['m_b_corr'].values
ceph   = dfm['CEPH_DIST'].values
is_cal = dfm['IS_CALIBRATOR'].values == 1
cid    = dfm['CID'].values

# ---------------------------------------------------------------
# 2. Repeated CIDs (same supernova observed by several surveys)
# ---------------------------------------------------------------
counts = dfm['CID'].value_counts()
dup_counts = counts[counts > 1]                       # CID -> how many times
groups = dfm.groupby('CID').indices                   # CID -> masked indices
dup_groups = {c: groups[c] for c in dup_counts.index} # only the repeated ones

print(f"Masked rows: {len(dfm)},  unique supernovae (CIDs): {dfm['CID'].nunique()}")
print(f"CIDs that appear more than once: {len(dup_counts)}")
print("How many CIDs appear 2, 3, 4... times:", dup_counts.value_counts().sort_index().to_dict())
for c, idx in list(dup_groups.items())[:5]:           # first few as an example
    print(f"  {c}: masked indices {idx.tolist()}, rows in .dat {orig_idx[idx].tolist()}")

# Map every row to its supernova: rows with the same CID get the same number.
# unique_cid[sn_index[i]] is the CID of row i.
unique_cid, sn_index = np.unique(cid, return_inverse=True)
n_sn = len(unique_cid)

# ---------------------------------------------------------------
# 3. Error columns and sigma2
# ---------------------------------------------------------------
err_raw  = dfm['m_b_corr_err_RAW'].values
err_bias = dfm['biasCorErr_m_b'].values
err_vpec = dfm['m_b_corr_err_VPEC'].values

sigma2 = err_raw**2 + err_bias**2 + err_vpec**2      # variance per row
sigma  = np.sqrt(sigma2)                             # standard deviation per row

# ---------------------------------------------------------------
# 4. Luminosity distance and mu1 (as before)
# ---------------------------------------------------------------
c = 299792.458  # km/s

def luminosity_distance(z, H0, Om=0.3):
    OL = 1.0 - Om
    E = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + OL)
    integral, _ = quad(E, 0, z)
    return (1 + z) * (c / H0) * integral

dL_times_H0 = np.array([luminosity_distance(zi, 1.0) for zi in z])

def mu1_model(H0):
    """mu from redshift: 5 log10(dL) + 25, one value per row."""
    return 5 * np.log10(dL_times_H0 / H0) + 25

# ---------------------------------------------------------------
# 5. Per-supernova Mb and mu drawn from Gaussians
# ---------------------------------------------------------------
def draw_Mb(Mb):
    """One Mb per row, drawn from N(Mb, sigma).
    Rows with the same CID share ONE draw (same supernova -> same Mb)."""
    sigma_sn = np.zeros(n_sn)
    np.maximum.at(sigma_sn, sn_index, sigma)   # one sigma per supernova (largest of its rows)
    Mb_sn = rng.normal(Mb, sigma_sn)           # one value per unique CID
    return Mb_sn[sn_index]                     # copy back to every row

def draw_mu2(H0):
    """mu per row drawn from N(mu1(H0), sigma)."""
    return rng.normal(mu1_model(H0), sigma)

# quick test
Mb_i  = draw_Mb(-19.25)
mu1_i = mu1_model(73.0)
mu2_i = draw_mu2(73.0)
print("Mb_i  shape:", Mb_i.shape, " first values:", np.round(Mb_i[:5], 3))
print("mu1_i shape:", mu1_i.shape, " first values:", np.round(mu1_i[:5], 3))
print("mu2_i shape:", mu2_i.shape, " first values:", np.round(mu2_i[:5], 3))
i0 = dup_groups[dup_counts.index[0]]
print(f"Check: rows of {dup_counts.index[0]} share Mb_i ->", np.round(Mb_i[i0], 4))
