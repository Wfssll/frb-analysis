"""Explicit CAMB precision settings used by the production calculations.

The original environment set these three boosts to 0.5 inside PyCCL's CAMB
adapter. PyCCL 3.2.1 does not expose them through extra_parameters. Apply the
same settings when CAMB creates its parameter object, without editing any
installed dependency. All other CAMB/PyCCL defaults remain unchanged.
"""
from functools import wraps
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import camb.model
import pyccl as ccl


def configure_camb_accuracy():
    constructor = camb.model.CAMBparams.__init__
    if getattr(constructor, "_frb_production_accuracy", False):
        return

    @wraps(constructor)
    def initialize(parameters, *args, **kwargs):
        constructor(parameters, *args, **kwargs)
        parameters.set_accuracy(AccuracyBoost=0.5, lSampleBoost=0.5,
                                lAccuracyBoost=0.5)

    initialize._frb_production_accuracy = True
    camb.model.CAMBparams.__init__ = initialize


def production_cosmology(*args, **kwargs):
    """Construct a CCL cosmology using the archived production precision."""
    configure_camb_accuracy()
    return ccl.Cosmology(*args, **kwargs)
