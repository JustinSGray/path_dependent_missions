from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from dymos import Phase, Trajectory

from path_dependent_missions.thermal_mission.thermal_mission_ode import ThermalMissionODE
from path_dependent_missions.utils.traj_plot import save_results, plot_results


def thermal_mission_trajectory(num_seg=5,
                               transcription_order=3,
                               meeting_altitude=20000.,
                               Q_env=0.,
                               Q_sink=0.,
                               Q_out=0.,
                               m_recirculated=0.,
                               opt_m_recirculated=False,
                               opt_m_burn=False,
                               opt_throttle=True,
                               engine_heat_coeff=0.,
                               pump_heat_coeff=0.,
                               T=None,
                               T_o=None,
                               opt_m=False,
                               m_initial=20.e3,
                               transcription='gauss-lobatto'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 50#00
    p.driver.opt_settings['Iterations limit'] = 5000000000000000
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-8
    p.driver.opt_settings['Verify level'] = -1
    p.driver.opt_settings['Linesearch tolerance'] = .1
    p.driver.opt_settings['Major step limit'] = .05
    p.driver.options['dynamic_simul_derivs'] = True

    traj = p.model.add_subsystem('traj', Trajectory())

    ascent = Phase(transcription, ode_class=ThermalMissionODE,
                        ode_init_kwargs={'engine_heat_coeff':engine_heat_coeff, 'pump_heat_coeff':pump_heat_coeff}, num_segments=num_seg,
                        transcription_order=transcription_order)

    ascent = traj.add_phase('ascent', ascent)

    ascent.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    ascent.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    ascent.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    ascent.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=5.0E-0, units='m/s')

    ascent.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    ascent.set_state_options('m', fix_initial=not opt_m, lower=15.e3, upper=80e3,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    ascent.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True)

    ascent.add_design_parameter('S', val=1., units='m**2', opt=False)

    if opt_throttle:
        ascent.add_control('throttle', val=1.0, lower=0.0, upper=1.0, opt=True, rate_continuity=True)
    else:
        ascent.add_design_parameter('throttle', val=1.0, opt=False)

    ascent.add_design_parameter('W0', val=10.5e3, opt=False, units='kg')

    ascent.add_boundary_constraint('h', loc='final', equals=meeting_altitude, scaler=1.0E-3, units='m')
    ascent.add_boundary_constraint('aero.mach', loc='final', equals=1., units=None)
    ascent.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    ascent.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    ascent.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    # ascent.add_path_constraint(name='time', upper=110.)

    # Minimize time at the end of the ascent
    ascent.add_objective('time', loc='final', ref=100.0)
    # ascent.add_objective('energy', loc='final', ref=100.0)
    # ascent.add_objective('m', loc='final', ref=-10000.0)

    # Set the state options for mass, temperature, and energy.
    ascent.set_state_options('T', fix_initial=True, ref=300, defect_scaler=1e-2)
    ascent.set_state_options('energy', fix_initial=True, ref=10e3, defect_scaler=1e-4)

    # Allow the optimizer to vary the fuel flow
    if opt_m_recirculated:
        ascent.add_control('m_recirculated', val=m_recirculated, lower=0., opt=True, rate_continuity=True, ref=20.)
    else:
        ascent.add_control('m_recirculated', val=m_recirculated, opt=False)

    ascent.add_design_parameter('Q_env', val=Q_env, opt=False)
    ascent.add_design_parameter('Q_sink', val=Q_sink, opt=False)
    ascent.add_design_parameter('Q_out', val=Q_out, opt=False)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    if opt_m_recirculated:
        ascent.add_path_constraint('m_flow', lower=0., upper=50., units='kg/s', ref=10.)

    if T is not None:
        ascent.add_path_constraint('T', upper=T, units='K')

    if T_o is not None:
        ascent.add_path_constraint('T_o', upper=T_o, units='K', ref=300.)




    cruise = Phase(transcription, ode_class=ThermalMissionODE,
                        ode_init_kwargs={'engine_heat_coeff':engine_heat_coeff, 'pump_heat_coeff':pump_heat_coeff}, num_segments=num_seg,
                        transcription_order=transcription_order)

    cruise = traj.add_phase('cruise', cruise)

    cruise.set_time_options(opt_initial=False, duration_bounds=(50, 1e4),
                           duration_ref=100.0)

    cruise.set_state_options('r', lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    cruise.set_state_options('h', lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    cruise.set_state_options('v', lower=10.0,
                            scaler=1.0E-2, defect_scaler=5.0E-0, units='m/s')

    cruise.set_state_options('gam', lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    cruise.set_state_options('m', lower=15.e3, upper=80e3,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    cruise.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True)

    cruise.add_design_parameter('S', val=1., units='m**2', opt=False)

    if opt_throttle:
        cruise.add_control('throttle', val=1.0, lower=0.0, upper=1.0, opt=True, rate_continuity=True)
    else:
        cruise.add_design_parameter('throttle', val=1.0, opt=False)

    cruise.add_design_parameter('W0', val=10.5e3, opt=False, units='kg')

    # Set the state options for mass, temperature, and energy.
    cruise.set_state_options('T', fix_initial=True, ref=300, defect_scaler=1e-2)
    cruise.set_state_options('energy', fix_initial=True, ref=10e3, defect_scaler=1e-4)

    # Allow the optimizer to vary the fuel flow
    if opt_m_recirculated:
        cruise.add_control('m_recirculated', val=m_recirculated, lower=0., opt=True, rate_continuity=True, ref=20.)
    else:
        cruise.add_control('m_recirculated', val=m_recirculated, opt=False)

    cruise.add_design_parameter('Q_env', val=Q_env, opt=False)
    cruise.add_design_parameter('Q_sink', val=Q_sink, opt=False)
    cruise.add_design_parameter('Q_out', val=Q_out, opt=False)


    cruise.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    cruise.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Link Phases (link time and all state variables)
    traj.link_phases(phases=['ascent', 'cruise'], vars=['*'])

    p.setup(mode='fwd', check=True)

    p['traj.ascent.t_initial'] = 0.0
    p['traj.ascent.t_duration'] = 200.
    p['traj.ascent.states:r'] = ascent.interpolate(ys=[0.0, 111319.54], nodes='state_input')
    p['traj.ascent.states:h'] = ascent.interpolate(ys=[100.0, meeting_altitude], nodes='state_input')
    # p['traj.ascent.states:h'][:] = 10000.

    p['traj.ascent.states:v'] = ascent.interpolate(ys=[135.964, 283.159], nodes='state_input')
    # p['traj.ascent.states:v'][:] = 200.
    p['traj.ascent.states:gam'] = ascent.interpolate(ys=[0.0, 0.0], nodes='state_input')
    p['traj.ascent.states:m'] = ascent.interpolate(ys=[m_initial, 20.e3], nodes='state_input')
    p['traj.ascent.controls:alpha'] = ascent.interpolate(ys=[1., 1.], nodes='control_input')

    # Give initial values for the ascent states, controls, and time
    p['traj.ascent.states:T'] = 310.


    p['traj.cruise.t_initial'] = 200
    p['traj.cruise.t_duration'] = 1000.
    p['traj.cruise.states:r'] = cruise.interpolate(ys=[111319.54, 2e6], nodes='state_input')
    p['traj.cruise.states:h'] = cruise.interpolate(ys=[meeting_altitude, meeting_altitude], nodes='state_input')
    # p['traj.cruise.states:h'][:] = 10000.

    p['traj.cruise.states:v'] = cruise.interpolate(ys=[283.159, 283.159], nodes='state_input')
    # p['traj.cruise.states:v'][:] = 200.
    p['traj.cruise.states:gam'] = cruise.interpolate(ys=[0.0, 0.0], nodes='state_input')
    p['traj.cruise.states:m'] = cruise.interpolate(ys=[14e3, 12.e3], nodes='state_input')
    p['traj.cruise.controls:alpha'] = cruise.interpolate(ys=[1., 1.], nodes='control_input')

    # Give initial values for the cruise states, controls, and time
    p['traj.cruise.states:T'] = 310.

    return p


if __name__ == '__main__':
    options = {
        'transcription' : 'gauss-lobatto',
        # 'transcription' : 'radau-ps',
        'num_seg' : 15,
        'transcription_order' : 3,
        'm_recirculated' : 0.,
        'opt_m_recirculated' : False,
        'Q_env' : 0.e3,
        'Q_sink' : 0.e3,
        'Q_out' : 0.e3,
        # 'T' : 315.,
        # 'T_o' : 330.,
        'm_initial' : 30.e3,
        'opt_throttle' : True,
        'opt_m' : False,
        'engine_heat_coeff' : 0.,
        }

    p = thermal_mission_trajectory(**options)

    # p.run_model()
    p.run_driver()

    save_results(p, 'new.pkl', options)
    plot_results(['new.pkl'], save_fig=False, list_to_plot=['h', 'aero.mach', 'm_fuel', 'T', 'T_o', 'm_flow', 'm_burn', 'm_recirculated', 'throttle'])
