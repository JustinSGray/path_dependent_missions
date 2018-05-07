from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

def plot_results(p, list_to_plot=[], sim_plot=True):
    """
    Helper function to perform explicit simulation at the optimized point
    and plot the results. Points are the collocation nodes and the solid
    lines are the explicit simulation results.

    Parameters
    ----------
    p : OpenMDAO Problem instance
        This problem instance should contain the optimized outputs from the
        energy minimization thermal problem.
    """

    if sim_plot:
        # This hides the print output of simulate
        sys.stdout = open(os.devnull, "w")
        out = p.model.phase.simulate(times=np.linspace(p['phase.t_initial'], p['phase.t_duration'], 100))
        sys.stdout = sys.__stdout__

    n_vars = len(list_to_plot)
    f, axarr = plt.subplots(n_vars, sharex=True)

    col_vals = OrderedDict()
    sim_vals = OrderedDict()

    col_vals['time'] = p.model.phase.get_values('time', nodes='all')
    if sim_plot:
        sim_vals['time'] = out.get_values('time')
    for i, name in enumerate(list_to_plot):
        col_vals[name] = p.model.phase.get_values(name, nodes='all')
        axarr[i].scatter(col_vals['time'], col_vals[name])

        if sim_plot:
            sim_vals[name] = out.get_values(name)
            axarr[i].plot(sim_vals['time'], sim_vals[name])

        axarr[i].set_ylabel('mass, kg')
        axarr[i].set_ylabel(name)

    axarr[-1].set_xlabel('time, sec')

    plt.show()
