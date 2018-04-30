from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian, SqliteRecorder

from dymos import Phase

from path_dependent_missions.simple_heat.tank_alone_ode import TankAloneODE
import numpy as np


def setup_energy_opt(num_seg, order, Q_env=0., Q_sink=0., Q_out=0., m_flow=0.1, m_burn=0., opt_m_flow=False, opt_m_burn=False):
    """
    Helper function to set up and return a problem instance for an energy minimization
    of a simple thermal system.

    Parameters
    ----------
    num_seg : int
        The number of ODE segments to use when discretizing the problem.
    order : int
        The order for the polynomial interpolation for the collocation methods.
    """

    # Instantiate the problem and set the optimizer
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 2000
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-7
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-7
    p.driver.opt_settings['Verify level'] = -1

    # Set up the phase for the defined ODE function, can be LGR or LGL
    phase = Phase('gauss-lobatto', ode_class=TankAloneODE,
                  ode_init_kwargs={}, num_segments=num_seg, transcription_order=order, compressed=True)

    # Do not allow the time to vary during the optimization
    phase.set_time_options(opt_initial=False, opt_duration=False)

    # Set the state options for mass, temperature, and energy.
    phase.set_state_options('m', lower=1., upper=10., fix_initial=False)
    phase.set_state_options('T', fix_initial=True)
    phase.set_state_options('energy', fix_initial=True)

    # Minimize the energy used to pump the fuel
    # phase.add_objective('energy', loc='final')
    phase.add_objective('m', loc='initial')

    # Allow the optimizer to vary the fuel flow
    if opt_m_flow:
        phase.add_control('m_flow', val=m_flow, dynamic=True, opt=True, rate_continuity=True)
    else:
        phase.add_control('m_flow', val=m_flow, dynamic=True, opt=False)

    if opt_m_burn:
        phase.add_control('m_burn', val=m_burn, dynamic=True, opt=True, rate_continuity=True)
    else:
        phase.add_control('m_burn', val=m_burn, dynamic=True, opt=False)

    phase.add_control('Q_env', val=Q_env, dynamic=False, opt=False)
    phase.add_control('Q_sink', val=Q_sink, dynamic=False, opt=False)
    phase.add_control('Q_out', val=Q_out, dynamic=False, opt=False)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    if opt_m_flow:
        phase.add_path_constraint('T', upper=1.)
        phase.add_path_constraint('m_flow_rate', upper=0.)
        phase.add_path_constraint('m_constraint', upper=0.)
        phase.add_path_constraint('m_flow', upper=3.)

    # Add the phase to the problem and set it up
    p.model.add_subsystem('phase', phase)
    p.driver.add_recorder(SqliteRecorder('out.db'))
    p.setup(check=True, force_alloc_complex=True, mode='fwd')

    # Give initial values for the phase states, controls, and time
    p['phase.states:m'] = 2.
    p['phase.states:T'] = 1.
    p['phase.states:energy'] = 0.
    p['phase.controls:m_burn'] = np.atleast_2d(np.linspace(0, 2., num_seg*order)).T
    p['phase.t_initial'] = 0.
    p['phase.t_duration'] = 10.

    p.set_solver_print(level=-1)

    return p

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
    energy2 = out.get_values('energy')

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(5, sharex=True)
    axarr[0].scatter(time, m)
    axarr[0].plot(time2, m2)
    axarr[0].set_ylabel('mass')

    axarr[1].scatter(time, T)
    axarr[1].plot(time2, T2)
    axarr[1].set_ylabel('temp')

    axarr[2].scatter(time, m_flow)
    axarr[2].plot(time2, m_flow2)
    axarr[2].set_ylabel('m_flow')

    axarr[3].scatter(time, m_burn)
    axarr[3].plot(time2, m_burn2)
    axarr[3].set_ylabel('m_burn')

    axarr[4].scatter(time, energy)
    axarr[4].plot(time2, energy2)
    axarr[4].set_ylabel('energy')

    axarr[-1].set_xlabel('time')

    plt.show()

if __name__ == '__main__':
    # p = setup_energy_opt(5, 3, Q_env=1., Q_sink=0., Q_out=0.)
    # p = setup_energy_opt(5, 3, Q_env=.8, Q_sink=0.8, Q_out=1.2, opt_m_burn=True, opt_m_flow=True)
    p = setup_energy_opt(15, 3, Q_env=1., Q_sink=1.5, Q_out=1.5, m_burn=.2, m_flow=1., opt_m_flow=True)
    # p = setup_energy_opt(5, 3, Q_env=1., Q_sink=0.5, Q_out=0., m_burn=1., m_flow=1.)


    p.run_driver()
    plot_results(p)
