# Data provenance and conventions

- `frb/chimefrbcat2.npy`: the project's input copy of CHIME/FRB catalog 2.
- `frb/chimefrbcat2_process.fits`: the 3,219-source processed sample used by
  the production fits. The `dm_exc_ne2001` column is **mean-centered**;
  `DMMEAN` in the FITS header stores the mean before centering. `dm_z.py`
  restores it before the DM likelihood. Do not treat this column as uncentered DM.
- `frb/chimefrbcat2_random_sky.fits`: the project's existing sky random catalog,
  retained with the FRB inputs; it is not needed to refit the supplied spectra.
- `frb/P_IGM.npz`: interpolation coefficients for the Macquart IGM scatter
  distribution. `P_IGM.py` contains the table-generation equations.
- `frb/localized_hosts_dm_eg.csv`: processed localized-host redshifts and DM
  corrections used as comparison points in the adopted DM–z figures. This
  table is the release input; rebuilding the upstream literature compilation
  is outside the fit-reproduction workflow.
  Workbook provenance uses the original workbook basename rather than a
  machine-specific absolute path; the workbook itself is not required for plotting.
- `galaxy/bgs_pz_dndz_gaussian_dz_0.01.txt`: stacked BGS photo-z PDFs; the
  header specifies Gaussian stacking and expected counts per interval.
- `galaxy/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt`: the project's DESI LRG
  North/South and combined tomographic redshift distributions.
- `angular power spectrum/cl_dm_catalog.npz`: measured DM auto- and galaxy–DM
  cross-spectra, joint covariance, bandpower windows, bin ordering, theoretical
  settings, and tracer n(z) arrays. Fits retain the first four BGS bins and all
  four LRG bins. The fifth BGS bin is not part of the production likelihood.

These files are supplied scientific inputs from the original project. Exact
original filenames and checksums are listed in `publication_files.json` at the
repository root. Catalog preprocessing notebooks and exploratory plots are
excluded. This release does not assert that those notebooks recreate the
published inputs unchanged.

CHIME/FRB and DESI observational products retain their original attribution and
distribution terms; the repository's MIT code license does not change them.
Consult the [CHIME/FRB data portal](https://www.chime-frb.ca/catalog) and
[DESI data portal](https://data.desi.lbl.gov/) for their documentation.
