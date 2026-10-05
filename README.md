# bayesian-statistics-experiment

Exercises and experiments in Bayesian statistics, starting with the hands-on
sessions of the **[AIPHY2: Bayesian Statistics](https://agenda.infn.it/event/51770/overview)**
school (Gran Sasso Science Institute, L'Aquila, 5–9 October 2026).

The first exercise infers the **Hubble constant H0** and the **absolute magnitude
of type Ia supernovae Mb** from the Pantheon+SH0ES data, using a Metropolis-Hastings
MCMC sampler written from scratch. Hamiltonian Monte Carlo comes next.

## Contents

```
bayesian-statistics-experiment/
├── scripts/
│   └── h0_mcmc.py      # Metropolis-Hastings fit of H0 and Mb
├── notebooks/          # Jupyter notebooks
├── data/               # Pantheon+SH0ES data (downloaded, not in git)
├── pyproject.toml      # project and dependencies (managed by uv)
└── uv.lock             # exact package versions
```

## Exercises

### 1. H0 and Mb with Metropolis-Hastings (`scripts/h0_mcmc.py`)

- **Data:** 1701 Pantheon+SH0ES supernovae, of which 354 are used:
  77 Cepheid calibrators and 277 Hubble-flow supernovae.
- **Model:** `m_b = Mb + mu`, where `mu` comes from Cepheid distances for the
  calibrators and from `5 log10(dL) + 25` (flat ΛCDM, Ωm = 0.3) for the
  Hubble flow.
- **Likelihood:** Gaussian, using the full statistical + systematic covariance matrix.
- **Prior:** uniform, H0 in [60, 80] km/s/Mpc and Mb in [-20, 20].
- **Sampler:** Metropolis-Hastings with a Gaussian proposal, 50,000 steps.
- **Output:** trace plots, corner plot and marginalized posteriors.

Result: **H0 ≈ 73.5 ± 1.0 km/s/Mpc**, **Mb ≈ −19.25 ± 0.03**.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and packages.

```bash
git clone git@github.com:jam-sadiq/bayesian-statistics-experiment.git
cd bayesian-statistics-experiment
uv sync          # creates .venv and installs everything from uv.lock
```

### Download the data

```bash
mkdir -p data && cd data
curl -L -o "Pantheon+SH0ES.dat" "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
curl -L -o "Pantheon+SH0ES_STAT+SYS.cov" "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
cd ..
```

## Usage

Run the script:

```bash
uv run python scripts/h0_mcmc.py
```

Or open Jupyter:

```bash
uv run jupyter lab
```

Add a new package:

```bash
uv add <package-name>
```

## Main packages

numpy, scipy, pandas, matplotlib, corner, emcee, jupyterlab

## Data reference

Pantheon+ and SH0ES data release:
https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_Data/4_DISTANCES_ANDCOVAR
