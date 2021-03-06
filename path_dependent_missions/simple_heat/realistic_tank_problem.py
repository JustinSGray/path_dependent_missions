from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, \
    SqliteRecorder

from dymos import Phase

from path_dependent_missions.simple_heat.tank_alone_ode import TankAloneODE
from path_dependent_missions.simple_heat.heat_plot_utils import plot_results

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
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-10
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-10
    p.driver.opt_settings['Verify level'] = -1

    # Set up the phase for the defined ODE function, can be LGR or LGL
    phase = Phase('gauss-lobatto', ode_class=TankAloneODE,
                  ode_init_kwargs={}, num_segments=num_seg, transcription_order=order, compressed=True)

    # Do not allow the time to vary during the optimization
    phase.set_time_options(opt_initial=False, opt_duration=False)

    # Set the state options for mass, temperature, and energy.
    phase.set_state_options('m', units='kg', lower=1e2, upper=1e5, fix_initial=True)
    phase.set_state_options('T', units='K', fix_initial=True)
    phase.set_state_options('energy', fix_initial=True)

    # Minimize the energy used to pump the fuel
    phase.add_objective('energy', loc='final', ref=10e2)
    # phase.add_objective('m', loc='initial')
    # phase.add_objective('time', loc='final')

    # Allow the optimizer to vary the fuel flow
    if opt_m_flow:
        phase.add_control('m_flow', val=m_flow, lower=0.01, opt=True, rate_continuity=True)
    else:
        phase.add_control('m_flow', val=m_flow, opt=False)

    if opt_m_burn:
        phase.add_control('m_burn', val=m_burn, lower=0.01, opt=True, rate_continuity=True)
    else:
        phase.add_control('m_burn', val=m_burn, opt=False)

    phase.add_control('Q_env', val=Q_env, dynamic=False, opt=False)
    phase.add_control('Q_sink', val=Q_sink, dynamic=False, opt=False)
    phase.add_control('Q_out', val=Q_out, dynamic=False, opt=False)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    if opt_m_flow:
        phase.add_path_constraint('T', lower=0.)
        phase.add_path_constraint('T', upper=333.)
        phase.add_path_constraint('T_o', lower=0., units='K')
        phase.add_path_constraint('T_o', upper=333., units='K')
        # phase.add_path_constraint('m_flow_rate', upper=0.)
        phase.add_path_constraint('m_recirculated', lower=0., units='kg/s')
        phase.add_path_constraint('m_flow', lower=0., upper=50.)

    # Add the phase to the problem and set it up
    p.model.add_subsystem('phase', phase)
    p.driver.add_recorder(SqliteRecorder('out.db'))
    p.setup(check=True, force_alloc_complex=True, mode='fwd')

    # Give initial values for the phase states, controls, and time
    p['phase.states:m'] = 6000
    p['phase.states:T'] = 300.
    p['phase.states:energy'] = 0.
    # p['phase.controls:m_burn'][:10] = np.atleast_2d(np.linspace(2., .5, num_seg*order)[:10]).T**2
    # p['phase.controls:m_burn'][10:] = 2.2
    p['phase.t_initial'] = 0.
    p['phase.t_duration'] = 9000

    p.set_solver_print(level=-1)

    return p

if __name__ == '__main__':

    p = setup_energy_opt(10, 3, Q_env=0, Q_sink=60.e3, Q_out=20.e3, m_burn=.5, m_flow=1.5, opt_m_flow=True)
    # p.run_model()
    p.run_driver()
    # p.check_partials(compact_print=True)
    plot_results(p)
