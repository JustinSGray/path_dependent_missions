import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from smt.surrogate_models import RMTB, RMTC, KRG


def get_data():
    # MN,  Alt (ft),  PC,  Fg (total gross thrust, lbf),  Ram (ram drag, lbf),  Fueltot (lbm/sec)
    data = np.loadtxt('good_output_flops')

    xt = data[:, :3]
    net_thrust = data[:, 3] - data[:, 4]
    yt = np.vstack((net_thrust, data[:, 5])).T

    # Scale inputs
    xt[:, 1] /= 1e4
    xt[:, 2] /= 1e2

    # Scale outputs
    yt[:, 0] /= 1e4
    yt[:, 1] /= 1e4

    xlimits = np.array([
        [0.0, 1.8],
        [0., 7.],
        [-.01, 1.01],
    ])

    return xt, yt, xlimits

def get_F110_interp():

    xt, yt, xlimits = get_data()

    interp = RMTB(xlimits=xlimits, num_ctrl_pts=15, order=4,
        approx_order=2, nonlinear_maxiter=40, solver_tolerance=1.e-20,
        # solver='lu', derivative_solver='lu',
        energy_weight=1.e-4, regularization_weight=0.e-18, extrapolate=False, print_global=True,
        data_dir='_smt_cache/',
    )

    # interp = KRG(theta0=[0.1]*3, data_dir='_smt_cache/')

    interp.set_training_values(xt, yt)
    interp.train()

    return interp


if __name__ == "__main__":
    from postprocessing.MultiView.MultiView import MultiView

    xt, yt, xlimits = get_data()
    interp = get_F110_interp()

    info = {'nx':3,
            'ny':2,
            'user_func':interp.predict_values,
            'resolution':150,
            'plot_size':12,
            'dimension_names':[
                'Mach number',
                'Altitude, 10k ft',
                'Throttle'],
            'bounds':xlimits.tolist(),
            'X_dimension':0,
            'Y_dimension':1,
            'scatter_points':[xt, yt],
            'dist_range': .1,
            'output_names':[
                'Net Thrust, 10k lbf',
                'Fuelburn, 10k lbm/sec',
            ]}

    # Initialize display parameters and draw GUI
    MultiView(info)
