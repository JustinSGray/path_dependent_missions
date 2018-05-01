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
    T_o = p.model.phase.get_values('T_o', nodes='all')
    m_flow = p.model.phase.get_values('m_flow', nodes='all')
    m_burn = p.model.phase.get_values('m_burn', nodes='all')
    m_recirculated = p.model.phase.get_values('m_recirculated', nodes='all')
    energy = p.model.phase.get_values('energy', nodes='all')
    Q_env = p.model.phase.get_values('Q_env', nodes='all')
    Q_sink = p.model.phase.get_values('Q_sink', nodes='all')
    Q_out = p.model.phase.get_values('Q_out', nodes='all')
    Q_tot_in = Q_env + Q_sink - m_recirculated * Q_out

    import sys
    import os

    # This hides the print output of simulate
    sys.stdout = open(os.devnull, "w")
    out = p.model.phase.simulate(times=np.linspace(0, 1, 100))
    sys.stdout = sys.__stdout__

    m2 = out.get_values('m')
    time2 = out.get_values('time')
    T2 = out.get_values('T')
    T_o2 = out.get_values('T_o')
    m_flow2 = out.get_values('m_flow')
    m_burn2 = out.get_values('m_burn')
    m_recirculated2 = out.get_values('m_recirculated')
    energy2 = out.get_values('energy')
    Q_env2 = out.get_values('Q_env')
    Q_sink2 = out.get_values('Q_sink')
    Q_out2 = out.get_values('Q_out')
    Q_tot_in2 = Q_env2 + Q_sink2 - m_recirculated2 * Q_out2

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(7, sharex=True)
    axarr[0].scatter(time, m)
    axarr[0].plot(time2, m2)
    axarr[0].set_ylabel('mass, kg')

    axarr[1].scatter(time, T)
    axarr[1].plot(time2, T2)
    axarr[1].set_ylabel('temp, K')

    axarr[2].scatter(time, T_o)
    axarr[2].plot(time2, T_o2)
    axarr[2].set_ylabel('temp HX1, K')

    axarr[3].scatter(time, m_flow)
    axarr[3].plot(time2, m_flow2)
    axarr[3].set_ylabel('m_flow, kg/s')

    axarr[4].scatter(time, m_burn)
    axarr[4].plot(time2, m_burn2)
    axarr[4].set_ylabel('m_burn, kg/s')

    axarr[5].scatter(time, m_recirculated)
    axarr[5].plot(time2, m_recirculated2)
    axarr[5].set_ylabel('m_recirculated, kg/s')

    axarr[6].scatter(time, Q_tot_in)
    axarr[6].plot(time2, Q_tot_in2)
    axarr[6].set_ylabel('Q_tot_in, W')

    # axarr[4].scatter(time, energy)
    # axarr[4].plot(time2, energy2)
    # axarr[4].set_ylabel('energy, J')

    axarr[-1].set_xlabel('time, sec')

    plt.show()
