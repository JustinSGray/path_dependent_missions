from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, \
    SqliteRecorder

from dymos import Phase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE
import numpy as np


def setup_energy_opt(num_seg, order, q_tank, q_hx1, q_hx2, opt_burn=False):
    """
    Helper function to set up and return a problem instance for an energy minimization
    of a simple thermal system.

    Parameters
    ----------
    num_seg : int
        The number of ODE segments to use when discretizing the problem.
    order : int
        The order for the polynomial interpolation for the collocation methods.
    q_tank : float
        The amount of inputted heat to the fuel tank. Positive is heat inputted
        to the fuel tank.
    q_hx1 : float
        A measure of the amount of heat added to the fuel at the first heat exchanger.
        This mimics the fuel handling the heat generated from a thermal load,
        such as avionicsc or air conditioners.
    q_hx2 : float
        A measure of the amount of heat added to the fuel at the second heat exchanger.
        This may be a negative number, which means that heat is being taken out
        of the fuel in some way.
    opt_burn : boolean
        If true, we allow the optimizer to control the amount of fuel burned
        in the system. This mimics the cost of the fuel needed in the plane
        to provide thrust.
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
    phase = Phase('gauss-lobatto', ode_class=SimpleHeatODE,
                  ode_init_kwargs={'q_tank': q_tank, 'q_hx1': q_hx1, 'q_hx2': q_hx2}, num_segments=num_seg, transcription_order=order)

    # Do not allow the time to vary during the optimization
    phase.set_time_options(opt_initial=False, opt_duration=False)

    # Set the state options for mass, temperature, and energy.
    phase.set_state_options('m', lower=1., upper=10., fix_initial=True)
    phase.set_state_options('T', fix_initial=True, defect_scaler=.01)
    phase.set_state_options('energy', fix_initial=True)

    # Minimize the energy used to pump the fuel
    phase.set_objective('energy', loc='final')

    # Allow the optimizer to vary the fuel flow
    phase.add_control('m_flow', opt=True, lower=0., upper=5., rate_continuity=True)

    # Optimize the burned fuel amount, if selected
    if opt_burn:
        phase.add_control('m_burn', opt=opt_burn, lower=.2, upper=5., dynamic=False)
    else:
        phase.add_control('m_burn', opt=opt_burn)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    phase.add_path_constraint('T', upper=1.)
    phase.add_path_constraint('m_flow_rate', upper=0.)
    phase.add_path_constraint('m_flow', upper=1.)
    phase.add_path_constraint('fuel_burner.m_recirculated', lower=0.)

    # Add the phase to the problem and set it up
    p.model.add_subsystem('phase', phase)
    p.driver.add_recorder(SqliteRecorder('out.db'))
    p.setup(check=True, force_alloc_complex=True)

    # Give initial values for the phase states, controls, and time
    p['phase.states:m'] = 10.
    p['phase.states:T'] = 1.
    p['phase.states:energy'] = 0.
    p['phase.controls:m_flow'] = .5
    p['phase.controls:m_burn'] = .1
    p['phase.t_initial'] = 0.
    p['phase.t_duration'] = 2.

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
    energy = p.model.phase.get_values('energy', nodes='all')

    out = p.model.phase.simulate(times=np.linspace(0, 1, 100))

    m2 = out.get_values('m')
    time2 = out.get_values('time')
    T2 = out.get_values('T')
    m_flow2 = out.get_values('m_flow')
    energy2 = out.get_values('energy')

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].scatter(time, m)
    axarr[0].plot(time2, m2)
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel('mass')

    axarr[1].scatter(time, T)
    axarr[1].plot(time2, T2)
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel('temp')

    axarr[2].scatter(time, m_flow)
    axarr[2].plot(time2, m_flow2)
    axarr[2].set_xlabel('time')
    axarr[2].set_ylabel('m_flow')

    axarr[3].scatter(time, energy)
    axarr[3].plot(time2, energy2)
    axarr[3].set_xlabel('time')
    axarr[3].set_ylabel('energy')

    plt.show()
