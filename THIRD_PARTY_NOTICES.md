# Third-party notices

The repository's MIT license covers its project code and release additions.
It does not replace third-party or observational-data licenses.

- `mymcmc` is the production local module. Its original source describes it as
  extracted and simplified from `qcosmc`; that attribution is retained in the
  module. This release does not include the unrelated example programs or chains.
- CHIME selection calculations use the public CHIME/FRB `chimefrb-selection`
  package, authored by Kyle McGregor and distributed under MIT according to its
  installed metadata. The pinned upstream dependency is installed separately.
  Its derived probability-interpolation table is included as scientific input.
- NumPy, SciPy, Matplotlib, Astropy, CCL, emcee, GetDist, dynesty, h5py, seaborn,
  and scikit-learn are external dependencies and retain their respective licenses.
- CHIME/FRB catalogs, DESI redshift distributions, localized-host literature
  inputs, and derived measurements retain their source attribution and terms.
- Times New Roman and Arial font files are not bundled. The source includes
  portable font fallbacks, so reproducing the data does not require these fonts.
