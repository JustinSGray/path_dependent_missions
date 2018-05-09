from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle


def save_results(p, filename, options={}, run_sim=True, list_to_save=['h', 'aero.mach', 'm_fuel', 'T', 'T_o', 'm_flow', 'm_burn', 'm_recirculated', 'throttle']):
    """
    Helper function to perform explicit simulation at the optimized point
    and save the results.

    Parameters
    ----------
    p : OpenMDAO Problem instance
        This problem instance should contain the optimized outputs from the
        energy minimization thermal problem.
    """

    if run_sim:
        # This hides the print output of simulate
        sys.stdout = open(os.devnull, "w")
        out = p.model.phase.simulate(times=np.linspace(p['phase.t_initial'], p['phase.t_duration'], 100))
        sys.stdout = sys.__stdout__

    n_vars = len(list_to_save)

    col_vals = OrderedDict()
    sim_vals = OrderedDict()

    col_vals['time'] = p.model.phase.get_values('time', nodes='all')
    if run_sim:
        sim_vals['time'] = out.get_values('time')
    for i, name in enumerate(list_to_save):
        try:
            col_vals[name] = p.model.phase.get_values(name, nodes='all')
        except:
            continue

        if run_sim:
            sim_vals[name] = out.get_values(name)

    big_dict = {'col_vals' : col_vals,
                'sim_vals' : sim_vals,
                'run_sim'  : run_sim,
                'options'  : options}

    with open(filename, 'wb') as f:
        pickle.dump(big_dict, f)

def plot_results(filename, save_fig=False, list_to_plot=['h', 'aero.mach', 'm_fuel', 'T', 'T_o', 'm_flow', 'm_burn', 'm_recirculated', 'throttle'], lines=[]):
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

    constraint_list = ['T', 'T_o']

    with open(filename, 'rb') as f:
        big_dict = pickle.load(f)

    col_vals = big_dict['col_vals']
    sim_vals = big_dict['sim_vals']
    sim_plot = big_dict['run_sim']
    options  = big_dict['options']

    n_vars = len(list_to_plot)
    f, axarr = plt.subplots(n_vars, sharex=True, figsize=(6, 10))

    for i, name in enumerate(list_to_plot):
        axarr[i].scatter(col_vals['time'], col_vals[name])

        if sim_plot:
            axarr[i].plot(sim_vals['time'], sim_vals[name])

        if name in options.keys() and name in constraint_list:
            axarr[i].axhline(y=options[name], color='r')

        axarr[i].set_ylabel(name)

    axarr[-1].set_xlabel('time, sec')

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename.split('.')[0]+'.pdf')
    else:
        plt.show()

if __name__ == "__main__":
    plot_results('test.pkl', save_fig=False, lines=[['T', 300.5]])
