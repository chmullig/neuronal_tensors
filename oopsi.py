import numpy as np
from scipy import signal
from pyfnnd import deconvolve, apply_all_cells
#from pyfnnd import plotting
import sys, os

infn = sys.argv[1]
inbase, _ = os.path.splitext(infn)
F = np.loadtxt(infn, delimiter=", ").T

n_best, c_best, LL, theta_best = apply_all_cells(
    F, dt=0.02, learn_theta=(0, 1, 1, 0, 0),
    spikes_tol=1E-6, params_tol=1E-6, norm_alpha=True, decimate=0, n_jobs=1
)

np.savez_compressed(infn + '.deconvolved.npz', n_best=n_best, c_best=c_best, LL=LL, theta_best=theta_best)

#plotting.ground_truth_1D(F, n_best, c_best, theta_best, n, c, theta, 0.02)

