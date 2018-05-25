from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle


def adjust_spines(ax = None, spines=['left'], off_spines=['top', 'right', 'bottom']):
    """ Function to shift the axes/spines so they have that offset
        Doumont look. """
    if ax == None:
        ax = plt.gca()

    # Loop over the spines in the axes and shift them
    for loc, spine in ax.spines.items():
        if loc in spines:
            # spine.set_position(('outward', 18))  # outward by 18 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    for spine in off_spines:
        ax.spines[spine].set_visible(False)

names = {}
names['h'] = 'altitude, m'
names['aero.mach'] = 'Mach'
names['m_fuel'] = 'fuel mass, kg'
names['T'] = 'tank T, K'
names['m_flow'] = '$\dot m_{flo}}$, kg/s'
names['m_recirculated'] = '$\dot m_{recirculated}$, kg/s'
names['m_burn'] = '$\dot m_{burn}$, kg/s'
names['throttle'] = 'throttle'

def save_results(p, filename, options={}, run_sim=True, list_to_save=['h', 'aero.mach', 'throttle', 'T', 'T_o', 'm_fuel', 'm_burn', 'm_recirculated']):
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

def plot_results(filenames, save_fig=None, list_to_plot=['h', 'aero.mach', 'throttle', 'T', 'm_fuel', 'm_burn', 'm_recirculated'], figsize=(6, 10), color_offset=0):
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

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if type(filenames) is not list:
        filenames = [filenames]

    n_vars = len(list_to_plot)
    f, axarr = plt.subplots(n_vars, sharex=False, figsize=figsize)

    max_times = []
    for j, filename in enumerate(filenames):
        with open(filename, 'rb') as f:
            big_dict = pickle.load(f)

        col_vals = big_dict['col_vals']
        sim_vals = big_dict['sim_vals']
        sim_plot = big_dict['run_sim']
        options  = big_dict['options']

        for i, name in enumerate(list_to_plot):
            # axarr[i].scatter(col_vals['time'], col_vals[name], color=colors[j+color_offset])

            if sim_plot:
                axarr[i].plot(sim_vals['time'], sim_vals[name], color=colors[j+color_offset], linewidth=3.)
        if sim_plot:
            max_times.append(np.max(sim_vals['time']))

    for i, name in enumerate(list_to_plot):
        if name in options.keys() and name in constraint_list:
            axarr[i].axhline(y=options[name], color='r')

        if name == 'throttle':
            axarr[i].set_ylim([-.05, 1.05])
            axarr[i].set_yticks([0., 1.])

        if name == 'aero.mach':
            axarr[i].set_ylim([-.05, 1.85])
            axarr[i].set_yticks([0., 1., 1.8])

        if name == 'T':
            axarr[i].set_ylim([309.8, 312.2])
            axarr[i].set_yticks([310, 312])

        if name == 'm_recirculated':
            axarr[i].set_ylim([-2, 52])
            axarr[i].set_yticks([0, 50])

        if name == 'm_burn':
            axarr[i].set_ylim([-2, 32])
            axarr[i].set_yticks([0, 30])

        if name == 'm_fuel':
            axarr[i].set_ylim([5000, 21000])
            axarr[i].set_yticks([5e3, 12.5e3, 20e3])

        if i < len(list_to_plot)-1:
            adjust_spines(axarr[i])
        else:
            adjust_spines(axarr[i], spines=['left', 'bottom'], off_spines=['top', 'right'])
            times = list(np.arange(0, np.min(max_times), 40))
            # times = [0.]
            times.extend(max_times)

            axarr[i].set_xticks(times)
            axarr[i].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if name in names.keys():
            axarr[i].set_ylabel(names[name])
        else:
            axarr[i].set_ylabel(name)

    axarr[-1].set_xlabel('time, sec')

    plt.tight_layout()

    if save_fig is not None:
        if save_fig:
            plt.savefig(filename.split('.')[0]+'.pdf')
    else:
        plt.show()

    return f, axarr

if __name__ == "__main__":
    plot_results('test.pkl', save_fig=False, lines=[['T', 300.5]])
