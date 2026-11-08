# -*- coding: utf-8 -*-
"""
mymcmc — Lightweight MCMC module extracted from qcosmc.

Usage::

    from mymcmc import MCMC_class, MCplot

    mc = MCMC_class(params, chi2, 'test', data_num=N, nwalkers=32)
    mc.runMC(steps=500, add_steps=200, max_steps=5000)
    # or: mc.run(steps=500, add_steps=200, max_steps=5000)

    pl = MCplot([['test_converged', 'test_converged']])
    pl.plot3D([1, 2, 3])
"""

from .mymcmc import (
    MCMC_class, MCplot, colors, lss, set_base_dir, run_dynesty,
    burn_in_steps, reprocess_directory, read_burn_in_record,
)

__all__ = [
    'MCMC_class', 'MCplot', 'colors', 'lss', 'set_base_dir', 'run_dynesty',
    'burn_in_steps', 'reprocess_directory', 'read_burn_in_record',
]
__version__ = '0.1.0'
