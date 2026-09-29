"""
device.py -- GPU-server device pinning.

IMPORTANT (read before assuming this makes the experiments GPU-bound):
Every algorithm in this repository (KOA and all six baselines) runs
through pymoo, which is a plain NumPy / CPU implementation -- there is
no tensor computation and nothing here executes on a GPU. Running
experiments does not require a GPU at all.

This module exists only so that, when this code is launched on a shared
GPU server / cluster node, it deterministically pins itself to a single
requested GPU index (device port) via the standard CUDA environment
variables, before any other library is imported -- so that on a
multi-tenant machine this job never silently grabs every visible GPU
from other users' jobs. Pinning does not accelerate anything here; it
is a good-citizen / cluster-policy setting requested for this codebase.

Usage
-----
    from device import pin_device
    pin_device(0)                 # pin to GPU 0 ("device port 0")

or from the command line, every run_*.py entry point in this repo
accepts `--gpu N` (default 0), which calls `pin_device(N)` as the very
first thing it does, before importing numpy/pymoo/etc.

Must be called BEFORE importing numpy, torch, tensorflow, numba, or any
GPU-aware library, since CUDA_VISIBLE_DEVICES is read once at process
start by those libraries' native/driver initialisation.
"""
import os


def pin_device(gpu_index: int = 0):
    """Pin this process to a single GPU device index ('device port').

    Safe to call even on a machine with no GPU / no CUDA driver at all --
    it only sets environment variables and never touches driver state
    itself, so pure-CPU runs (the normal case for this repository) are
    completely unaffected.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Also pin common alternative-backend device selectors some clusters
    # key off of, harmless no-ops if those libraries are not installed.
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(gpu_index))
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", str(gpu_index))
    return gpu_index
