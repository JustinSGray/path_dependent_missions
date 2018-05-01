from __future__ import print_function, division, absolute_import
import numpy as np

def plot_results(p):
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
    m = p.model.phase.get_values('m', nodes='all')
    time = p.model.phase.get_values('time', nodes='all')
    T = p.model.phase.get_values('T', nodes='all')
    m_flow = p.model.phase.get_values('m_flow', nodes='all')
    m_burn = p.model.phase.get_values('m_burn', nodes='all')
    m_recirculated = p.model.phase.get_values('m_recirculated', nodes='all')
    energy = p.model.phase.get_values('energy', nodes='all')

    import sys
    import os

    # This hides the print output of simulate
    sys.stdout = open(os.devnull, "w")
    out = p.model.phase.simulate(times=np.linspace(0, 1, 100))
    sys.stdout = sys.__stdout__

    m2 = out.get_values('m')
    time2 = out.get_values('time')
    T2 = out.get_values('T')
    m_flow2 = out.get_values('m_flow')
    m_burn2 = out.get_values('m_burn')
    m_recirculated2 = out.get_values('m_recirculated')
    energy2 = out.get_values('energy')

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(5, sharex=True)
    axarr[0].scatter(time, m)
    axarr[0].plot(time2, m2)
    axarr[0].set_ylabel('mass, kg')

    axarr[1].scatter(time, T)
    axarr[1].plot(time2, T2)
    axarr[1].set_ylabel('temp, K')

    axarr[2].scatter(time, m_flow)
    axarr[2].plot(time2, m_flow2)
    axarr[2].set_ylabel('m_flow, kg/s')

    axarr[3].scatter(time, m_burn)
    axarr[3].plot(time2, m_burn2)
    axarr[3].set_ylabel('m_burn, kg/s')

    axarr[4].scatter(time, m_recirculated)
    axarr[4].plot(time2, m_recirculated2)
    axarr[4].set_ylabel('m_recirculated, kg/s')

    # axarr[4].scatter(time, energy)
    # axarr[4].plot(time2, energy2)
    # axarr[4].set_ylabel('energy, J')

    axarr[-1].set_xlabel('time, sec')

    plt.show()
