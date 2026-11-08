# FRB analysis

Angular power spectra, dispersion measures, redshift distributions, and delay-time
inference for CHIME fast radio bursts and BGS/LRG galaxies.

This repository contains the production inference code, measured inputs, cleaned
posterior samples, nested-sampling evidence, eight scientific robustness cases,
and the fifteen adopted figures. Development experiments, software tests,
scheduler scripts, temporary chains, checkpoints, and diagnostic images are
excluded.

## Install

Use Linux and Python 3.10. The reference scientific package versions are pinned:

```bash
conda env create -f environment.yml
conda activate frb-analysis
```

Run commands from the repository root. The local `mymcmc` package is included;
it does not need a separate installation. CCL uses compiled libraries, so the
conda-forge environment is preferred over a source-only pip installation.
Matplotlib uses a headless backend and falls back to DejaVu fonts; regenerated
figures can have different text spacing from the archived publication PDFs.

`cosmology.py` explicitly preserves the production CAMB precision settings
(`AccuracyBoost`, `lSampleBoost`, and `lAccuracyBoost`, each 0.5). The original
environment implemented these inside its PyCCL installation; this repository
applies them in process, so installing standard upstream packages is sufficient.

## Contents

| Directory | Contents |
| --- | --- |
| `result/` | Historical production results: `cl`, `dmz`, `joint`; Gaussian, log-normal, power-law, and no-delay SFH models |
| `null/` | Archived DM-permutation measurements for the scientific null figure |
| `robust/` | Eight production robustness cases, their scientific configurations, selection functions, and cleaned chains |
| `data/frb/` | FRB catalogs, IGM interpolation table, and processed localized-host data |
| `data/galaxy/` | BGS and LRG redshift-distribution tables |
| `data/angular power spectrum/` | Measured spectra, covariance, bandpower windows, and tracer redshift distributions |
| `CHIME_selection/` | Detection probabilities, sightline draw, interpolation table, and generation code |
| `picture/` | Exactly fifteen adopted PDFs, PNG counterparts, and necessary numerical summaries |
| `mymcmc/` | Production emcee/dynesty sampling implementation |

## Reproduce the figures

The quickest route uses the published chains and numerical summaries:

```bash
python picture.py --figure all
python picture.py --figure pz
```

To keep the archived publication images intact, write regenerated outputs to a
separate location:

```bash
python picture.py --figure all --output-dir runs/figures
```

The adopted figure stems are `bias`, `contour_map_gauss`,
`contour_map_lognormal`, `contour_map_powerlaw`, `cov_matrix`, `data`, `dmz`,
`nz_galaxy_test`, `pz`, `P_det_test2`, `test_bias`, `test_dmz`, `test_null`,
`test_pz_mean`, and `test_constant_mean_clean`. The `test` spelling in these
adopted scientific figures is intentional.

`pz` and `test_pz_mean` display pointwise posterior means and population standard
deviations. They are not interchangeable with median/quantile credible bands.
DM–z figures show conditional probability densities and predictive quantiles.
The selection illustration uses Gaussian delays of 1, 3, and 5 Gyr with a fixed
1.5 Gyr width; these illustrate the selection effect rather than fitted centers.
The null figure uses the mean and sample standard deviation of 1,000 DM
permutations (seed 42), with fixed sky positions.

## Regenerate the CHIME selection function

```bash
python -m CHIME_selection.simulate_frb_data --output runs/selection/pdet_chime.npz
```

The default uses the archived sightline draw and probability-interpolation
table. It recomputes the energy/redshift quadrature rather than copying saved
probabilities. Rebuild the interpolation table from the pinned upstream model:

```bash
python -m CHIME_selection.simulate_frb_data --rebuild-table --output runs/selection/rebuilt.npz
```

See [the generation notes](CHIME_selection/README.md) for parameters and inputs.
The published reference selection file is protected from accidental overwriting.

## Rerun parameter inference

One baseline fit, all twelve fits, or one scientific robustness case:

```bash
python run_analysis.py --mode cl --model gauss --cores 4
python run_analysis.py --mode all --model all --cores 16
python run_analysis.py --case alpha_17 --cores 4
python run_analysis.py --case all --cores 16
```

Nested sampling for evidence uses the same model and likelihood:

```bash
python run_analysis.py --mode all --model all --sampler dynesty --cores 16
```

Outputs go to `runs/` and never overwrite the historical production results.
Choose another location with `--output`. Production emcee defaults are 32
walkers, 50,000 initial steps, up to 100,000 steps, and 30% burn-in discarded
along each walker's time axis. The stored `*_converged.npy` samples have already
had burn-in removed: **do not discard another 30%**. Historical stopping details
are recorded in each `chains/burn_in.json`. Short runs can exercise execution
but are not scientifically converged replacements.

These are CPU-intensive calculations. Use multiple cores and allow substantial
time for production MCMC and DM–z posterior-predictive calculations. Stochastic
reruns are assessed by convergence and agreement of posterior summaries, not
byte-identical samples. The supplied cleaned chains avoid the need to rerun
sampling just to inspect the main results.

For an execution-only short run, skip the optional preliminary optimizer:

```bash
python run_analysis.py --mode cl --model gauss --steps 2 --max-steps 2 --no-optimize
```

To recompute posterior summaries from newly sampled chains, select the new
result roots and use `--recompute` (the default roots are the published results):

```bash
RESULT_ROOT="$PWD/runs" ROBUST_ROOT="$PWD/runs/robust" python picture.py --figure all --recompute --output-dir runs/figures
```

Both the baseline and robustness runs must exist for that full command. Cached
scientific summaries are used by default when redrawing the historical results.

## Reproducibility boundary and provenance

The starting point is the supplied measured angular-spectrum archive, FRB
catalog, and redshift-distribution tables. Raw BGS/LRG sky and random catalogs
are not supplied; re-estimating the measured spectra or regenerating the null
permutations from those catalogs is outside this release. The archived null
measurements can be independently redrawn. Theory fitting uses the archived
NaMaster windows and does not require NaMaster to be installed.

Historical `run_manifest.npz` files are retained unchanged as records of the
original runs. Their old source/path fingerprints are not certificates for the
reorganized source. `publication_files.json` records the original-to-release
mapping and original file checksums; `SHA256SUMS` records published contents.
The path changes, extracted core code, and figure selection are deliberate
release transformations. The source project remains separate.

See [data provenance](data/README.md), [third-party notices](THIRD_PARTY_NOTICES.md),
and [reproduction notes](REPRODUCIBILITY.md). Code is distributed under the MIT
license; observational data and third-party materials retain their source terms.
