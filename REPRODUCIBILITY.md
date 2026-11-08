# Reproduction notes

This release was assembled separately from the original working project.
Historical samples and scientific inputs are preserved, with source mappings
in `publication_files.json` and release-content checksums in `SHA256SUMS`.

The release preparation verified:

- All twenty emcee chain archives have finite samples, consistent parameter
  dimensions, and their recorded 30% burn-in convention. Twelve nested-sampling
  archives and their evidence summaries are included.
- The original and reorganized implementations give matching parameter order,
  initial likelihood components, and angular-spectrum model vectors for all
  twelve baseline configurations and eight robustness cases.
- All fifteen adopted PDF/PNG pairs can be regenerated using only the release
  inputs and reference scientific dependencies. Representative figures were
  inspected visually. The measured-spectrum figure was also regenerated with
  fresh theory curves rather than the archived curve cache.
- The selection generator agrees with the production detection probabilities
  within the documented numerical tolerance, both with the archived table and
  with a freshly rebuilt upstream interpolation table. The largest absolute
  difference was approximately 3e-12.
- Two-step, 32-walker execution checks completed for Gaussian-delay `cl`,
  `dmz`, and `joint` runs. These check the sampling/output path and do not
  establish convergence. The production-length chains were not resampled as
  part of release preparation.

Verification scripts, logs, short chains, and regenerated inspection images
are excluded from this repository. These notes summarize the checks rather
than distributing their working artifacts.

A separate Linux environment was successfully installed from `environment.yml`.
All twenty likelihood/model-vector configurations were independently compared
with the original implementation's reference values in that clean environment,
and all fifteen adopted PDF/PNG pairs were redrawn there. The reference
environment checks and clean-installation checks were performed separately.

The original environment had a local PyCCL adapter change setting CAMB's three
accuracy boosts to 0.5. This setting is now explicit in `cosmology.py`, applied
to CAMB parameter construction in process without modifying installed packages.
Simply pinning package version numbers would otherwise miss this setting.
The original installation also disabled two massive-neutrino hierarchy checks;
that unrelated change is not carried over because these calculations use the
default massless-neutrino cosmology.

Raw-galaxy-catalog spectrum estimation and fresh null-permutation measurements
are outside the release boundary. Localized-host compilation and original
Galactic-DM corrections are supplied as processed inputs. Full DM–z predictive
recomputation may be substantially more expensive than cached redrawing.
