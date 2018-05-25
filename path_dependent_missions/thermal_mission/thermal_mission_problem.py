from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from path_dependent_missions.thermal_mission.thermal_mission_ode import ThermalMissionODE
from path_dependent_missions.utils.gen_mission_plot import save_results, plot_results


def thermal_mission_problem(num_seg=5, transcription_order=3, meeting_altitude=20000., Q_env=0., Q_sink=0., Q_out=0., m_recirculated=0., opt_m_recirculated=False, opt_m_burn=False, opt_throttle=True, engine_heat_coeff=0., pump_heat_coeff=0., T=None, T_o=None, opt_m=False, m_initial=20.e3, transcription='gauss-lobatto'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 5000
    p.driver.opt_settings['Iterations limit'] = 5000000000000000
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-8
    p.driver.opt_settings['Verify level'] = -1
    p.driver.opt_settings['Linesearch tolerance'] = .1
    p.driver.opt_settings['Major step limit'] = .1
    p.driver.options['dynamic_simul_derivs'] = True
    p.driver.options['dynamic_simul_derivs_repeats'] = 5

    phase = Phase(transcription, ode_class=ThermalMissionODE,
                        ode_init_kwargs={'engine_heat_coeff':engine_heat_coeff, 'pump_heat_coeff':pump_heat_coeff}, num_segments=num_seg,
                        transcription_order=transcription_order)

    p.model.add_subsystem('phase', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=5.0E-0, units='m/s')

    phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    phase.set_state_options('m', fix_initial=not opt_m, lower=15.e3, upper=80e3,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=1., units='m**2', dynamic=False, opt=False)

    if opt_throttle:
        phase.add_control('throttle', val=1.0, lower=0.0, upper=1.0, dynamic=True, opt=True, rate_continuity=True)
    else:
        phase.add_control('throttle', val=1.0, dynamic=False, opt=False)

    phase.add_control('W0', val=10.5e3, dynamic=False, opt=False, units='kg')

    phase.add_boundary_constraint('h', loc='final', equals=meeting_altitude, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1., units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    # phase.add_path_constraint(name='time', upper=110.)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=100.0)
    # phase.add_objective('energy', loc='final', ref=100.0)
    # phase.add_objective('m', loc='final', ref=-10000.0)

    # Set the state options for mass, temperature, and energy.
    phase.set_state_options('T', fix_initial=True, ref=300, defect_scaler=1e-2)
    phase.set_state_options('energy', fix_initial=True, ref=10e3, defect_scaler=1e-4)

    # Allow the optimizer to vary the fuel flow
    if opt_m_recirculated:
        phase.add_control('m_recirculated', val=m_recirculated, lower=0., dynamic=True, opt=True, rate_continuity=True, ref=20.)
    else:
        phase.add_control('m_recirculated', val=m_recirculated, dynamic=True, opt=False)

    phase.add_control('Q_env', val=Q_env, dynamic=False, opt=False)
    phase.add_control('Q_sink', val=Q_sink, dynamic=False, opt=False)
    phase.add_control('Q_out', val=Q_out, dynamic=False, opt=False)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    if opt_m_recirculated:
        phase.add_path_constraint('m_flow', lower=0., upper=50., units='kg/s', ref=10.)

    if T is not None:
        phase.add_path_constraint('T', upper=T, units='K')

    if T_o is not None:
        phase.add_path_constraint('T_o', upper=T_o, units='K', ref=300.)

    # phase.add_path_constraint('m_flow', lower=0., upper=20., units='kg/s', ref=10.)

    p.setup(mode='fwd', check=True)

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 200.
    p['phase.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['phase.states:h'] = phase.interpolate(ys=[100.0, meeting_altitude], nodes='disc')
    # p['phase.states:h'][:] = 10000.

    p['phase.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
    # p['phase.states:v'][:] = 200.
    p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase.states:m'] = phase.interpolate(ys=[m_initial, 12.e3], nodes='disc')
    p['phase.controls:alpha'] = phase.interpolate(ys=[1., 1.], nodes='all')

    # Give initial values for the phase states, controls, and time
    p['phase.states:T'] = 310.

    return p


if __name__ == '__main__':
    options = {
        'transcription' : 'gauss-lobatto',
        'num_seg' : 15,
        'transcription_order' : 3,
        'm_recirculated' : 0.,
        'opt_m_recirculated' : False,
        'Q_env' : 300.e3,
        'Q_sink' : 40.e3,
        'Q_out' : 0.e3,
        # 'T' : 315.,
        # 'T_o' : 330.,
        'm_initial' : 30.e3,
        'opt_throttle' : True,
        'opt_m' : False,
        'engine_heat_coeff' : 0.,
        }

    p = thermal_mission_problem(**options)
    p.run_driver()
    # p.run_model()

    save_results(p, 'new.pkl', options)
    plot_results(['new.pkl'], save_fig=False, list_to_plot=['h', 'aero.mach', 'm_fuel', 'T', 'T_o', 'm_flow', 'm_burn', 'm_recirculated', 'throttle'])
