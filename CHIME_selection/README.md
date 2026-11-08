# CHIME detection-probability generation

`pdet_chime.npz` is the production DM-dependent detection function. The
generator evaluates selection probability on a shared log-energy grid and
averages over archived catalog sightlines. It combines Galactic DM, Galactic
halo DM, the cosmological IGM mean, and observer-frame host DM.

- Cosmology: Omega_m=0.3111, Omega_b=0.04897, h=0.6766, sigma8=0.81, n_s=0.96.
- Baryon/ionization factors: f_IGM=0.93, X_e=0.875.
- Redshift: 500 bin centers on [0,5], starting at 0.005.
- Energy: 1,000 logarithmic samples from 1e38 to 1e42 erg.
- Energy distribution: (E/E_cut)^(-1.8) exp(-E/E_cut), E_cut=3e41 erg.
- Spectral bandwidth: 400 MHz; source-frame host DM: 100 pc cm^-3.
- Sightlines: 1,000 catalog positions, original random seed 42.
- Selection model: predictors fluence and DM, degree 5, S/N cut 12,
  sidelobes excluded with cut 5, reweighting enabled.
- Probability table: 400 by 400 points in log fluence and log DM, with
  the production boundary/interpolation conventions retained in `selection_model.py`.

`sightlines.npz` stores only the shared z/energy grids, sampled RA/Dec, and
precomputed Galactic/halo DM corrections. These corrections came from the
original electron-density calculations (NE2001 and the analytic YT2020 halo);
they are reused to reproduce the actual sightline draw without environment-
dependent reruns of those calculations. No saved detection probabilities are
used as generator inputs.

`selection_table.npz` stores the upstream selection interpolation input.
`--rebuild-table` regenerates it in memory using
[CHIMEFRB/chimefrb-selection](https://github.com/CHIMEFRB/chimefrb-selection),
commit `b20bb28438e2bcd4bfed5d5e51fa96a20eb7a464` (package version 0.1.0).
The default archived-table route does not need that package at runtime.

The command accepts `--f-igm`, `--alpha`, `--e-cut`, and `--dm-host` for
the published scientific robustness variations. Case-specific production
selection functions are preserved below `robust/`.
