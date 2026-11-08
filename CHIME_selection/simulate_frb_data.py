"""Regenerate the adopted CHIME P_det from archived Monte Carlo sightlines.

Run as: python -m CHIME_selection.simulate_frb_data --output runs/selection.npz
"""
import argparse
from pathlib import Path
import numpy as np
import pyccl as ccl
from . import selection_model as model

HERE = Path(__file__).resolve().parent
Z_MIN = 0.0
Z_MAX = 5.0

def generate(output, *, f_igm=0.93, alpha=1.8, e_cut=3e41, dm_host=100.0,
             rebuild_table=False):
    with np.load(HERE / "sightlines.npz", allow_pickle=False) as saved:
        data = dict(saved)
    if rebuild_table:
        import chimefrb_selection as cfsf
        sf = cfsf.SelectionFunction(predictor_names=["fluence", "dm"],
                degree=5, snr_cut=12.0, exclude_sidelobes=True,
                sidelobe_cut=5.0, reweighted=True)
        f_grid, dm_grid, p_tab = model.tabulate_selection(sf, *model.training_domain(sf))
    else:
        with np.load(HERE / "selection_table.npz", allow_pickle=False) as saved:
            f_grid, dm_grid, p_tab = (saved[k] for k in ("f_grid", "dm_grid", "p_tab"))
    model.F_IGM = float(f_igm)
    z, energy = data["z"], data["E"]
    dm_igm = model.build_dm_igm(z)
    weights, wsum = model.energy_weight(energy, alpha, e_cut)
    distance = np.maximum(ccl.comoving_radial_distance(model.COSMO, 1/(1+z)), 1e-8) * (1+z)
    raw = model.pdet_for_host(z, energy, distance, dm_igm,
            data["dm_mw"] + data["dm_halo"], f_grid, dm_grid, p_tab, dm_host,
            [("case", weights, wsum)])["case"]
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("Invalid detection probabilities")
    data.update(p_raw=raw, p_det=raw, w=weights, dm_igm=dm_igm)
    if (f_igm, alpha, e_cut, dm_host) != (0.93,1.8,3e41,100.0):
        data.update(f_igm=np.asarray(f_igm),alpha=np.asarray(alpha),
                    e_cut=np.asarray(e_cut),selection_dm_host=np.asarray(dm_host))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.resolve() == (HERE / "pdet_chime.npz").resolve():
        raise ValueError("Choose a new output path to preserve the published selection")
    np.savez(output, **data)
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE.parent / "runs/selection/pdet_chime.npz")
    p.add_argument("--f-igm", type=float, default=0.93)
    p.add_argument("--alpha", type=float, default=1.8)
    p.add_argument("--e-cut", type=float, default=3e41)
    p.add_argument("--dm-host", type=float, default=100.)
    p.add_argument("--rebuild-table", action="store_true")
    a=p.parse_args()
    generate(a.output,f_igm=a.f_igm,alpha=a.alpha,e_cut=a.e_cut,
             dm_host=a.dm_host,rebuild_table=a.rebuild_table)
    print("Saved:", a.output)

if __name__ == "__main__":
    main()
