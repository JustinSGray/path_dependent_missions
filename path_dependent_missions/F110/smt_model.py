import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from smt.surrogate_models import RMTB

# MN,  Alt,  PC,  Fg (total gross thrust, lbf),  Ram (ram drag, lbf),  Fueltot (lbm/sec)
data = np.loadtxt('good_output_flops')

print(data.shape)
xt = data[:, :3]
yt = data[:, 3:]

# Scale inputs
xt[:, 1] /= 1e4
xt[:, 2] /= 1e2

# Scale outputs
yt[:, 0] /= 1e4
yt[:, 1] /= 1e4
yt[:, 2] /= 1e4

xlimits = np.array([
    [0.0, 1.8],
    [0., 7.],
    [-.01, 1.01],
])

interp = RMTB(xlimits=xlimits, num_ctrl_pts=15, order=4,
    approx_order=2, nonlinear_maxiter=0, solver_tolerance=1.e-20,
    # solver='lu', derivative_solver='lu',
    energy_weight=1.e-4, regularization_weight=0.e-20, extrapolate=False, print_global=True,
    data_dir='_smt_cache/',
)

interp.set_training_values(xt, yt)
interp.train()

from postprocessing.MultiView.MultiView import MultiView
import pickle
import scipy.interpolate

info = {'nx':3,
        'ny':3,
        'user_func':interp.predict_values,
        'resolution':60,
        'plot_size':8,
        'dimension_names':[
            'Mach number',
            'Altitude',
            'Throttle'],
        'bounds':xlimits.tolist(),
        'X_dimension':0,
        'Y_dimension':1,
        'scatter_points':[xt, yt],
        'dist_range': .1,
        'output_names':[
            'Thrust',
            'Drag',
            'Fuelburn',
        ]}

# Initialize display parameters and draw GUI
MultiView(info)
