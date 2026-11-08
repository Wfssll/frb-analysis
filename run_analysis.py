"""Run baseline fits or the eight scientific robustness cases locally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MODES=("cl","dmz","joint")
MODELS={"gauss":"gauss","lognormal":"lognormal","powerlaw":"power-law","sfh":"SFH"}
CASES=("ell_loc_free","ell_loc_500","fd_084","alpha_17","alpha_19","ecut_1e41","ecut_5e41","dm_host_150")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=(*MODES,"all"),default="cl")
    p.add_argument("--model",choices=(*MODELS,"all"),default="gauss")
    p.add_argument("--case",choices=(*CASES,"all"))
    p.add_argument("--sampler",choices=("emcee","dynesty"),default="emcee")
    p.add_argument("--cores",type=int,default=1)
    p.add_argument("--steps",type=int,default=50000)
    p.add_argument("--max-steps",type=int,default=100000)
    p.add_argument("--no-optimize",action="store_true",
                   help="Start walkers near the configured fiducial point without Nelder-Mead")
    p.add_argument("--output",type=Path,default=ROOT / "runs")
    a=p.parse_args()
    if a.cores<1 or a.steps<1 or a.max_steps<a.steps:
        p.error("Require positive cores and 1 <= steps <= max-steps")
    if a.case and a.sampler != "emcee":p.error("Published robustness cases use emcee")
    base_env=dict(os.environ)
    for key in list(base_env):
        if key.startswith(("MCMC_","MYMCMC_","CL_","DYNESTY_")) or key in ("SKIP_MCMC","RUN_DYNESTY","DELAY_MODEL","IGM_MODEL","PDET_FILE","ASTRA_ROOT","RESULT_ROOT","SFH_YEAR"):
            base_env.pop(key,None)
    base_env.update(IGM_MODEL="macquart",SFH_YEAR="2017",MPLBACKEND="Agg",
        MCMC_BGS_KEEP="0,1,2,3",MCMC_LRG_KEEP="0,1,2,3",MCMC_CL_BANDPOWER="1",
        MCMC_CL_BANDPOWER_NELL="24",MCMC_CL_BANDPOWER_GRID="log",MCMC_BURN_IN_FRAC="0.3",
        MCMC_FIT_K_CUT="1",MCMC_FIT_B_FRB="1",MCMC_FIT_DM_HOST="1",
        MCMC_FIT_BETA_FRB="0",MCMC_FIT_B_E="0",
        CL_DATA_FILE=str(ROOT / "data/angular power spectrum/cl_dm_catalog.npz"),
        PDET_FILE=str(ROOT / "CHIME_selection/pdet_chime.npz"))
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        base_env[key]="1"
    output=a.output.resolve()
    if output == ROOT / "result" or ROOT / "result" in output.parents or output == ROOT / "robust" or ROOT / "robust" in output.parents:
        p.error("Choose a separate output directory to preserve published chains")
    tasks=[("cl","gauss",c) for c in (CASES if a.case=="all" else (a.case,))] if a.case else [(m,d,None) for m in (MODES if a.mode=="all" else (a.mode,)) for d in (MODELS if a.model=="all" else (a.model,))]
    for mode,delay,case in tasks:
        env=dict(base_env)
        dest=output / mode / MODELS[delay]
        if case:
            cfg=json.loads((ROOT / "robust" / case / "config.json").read_text())
            env.update(cfg["environment"])
            env["CL_DATA_FILE"]=base_env["CL_DATA_FILE"]
            if cfg.get("ell_loc") == 500:
                import numpy as np
                with np.load(env["CL_DATA_FILE"],allow_pickle=False) as saved:catalog=dict(saved)
                catalog["ell_loc"]=np.asarray(500.,dtype=catalog["ell_loc"].dtype)
                path=output / "robust" / case / "cl_dm_catalog.npz"
                path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**catalog)
                env["CL_DATA_FILE"]=str(path)
            env["PDET_FILE"]=str(ROOT / cfg["environment"]["PDET_FILE"])
            dest=output / "robust" / case / "cl/gauss"
        if a.sampler=="dynesty":
            dest=dest / "dynesty";env["RUN_DYNESTY"]="1"
        else:env.pop("RUN_DYNESTY",None)
        env.update(MCMC_MODE=mode,DELAY_MODEL=delay,MYMCMC_NCORES=str(a.cores),
            MCMC_STEPS=str(a.steps),MCMC_MAX_STEPS=str(a.max_steps),
            MCMC_ADD_STEPS="10000",MCMC_OUT_DIR=str(dest),PYTHONUNBUFFERED="1")
        env["MCMC_OPTIMIZE"]="0" if a.no_optimize else "1"
        for key in ("CL_DATA_FILE","PDET_FILE"):
            if not Path(env[key]).is_file():p.error("Missing input: "+env[key])
        print("Running",mode,delay,case or "baseline","->",dest,flush=True)
        subprocess.run([sys.executable,"-u",str(ROOT / "mcmc_frb_redshift.py"),"--mode",mode],cwd=ROOT,env=env,check=True)

if __name__ == "__main__":main()
