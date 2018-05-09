from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian, SqliteRecorder

from dymos import Phase

from thermal_mission_ode import ThermalMissionODE
from path_dependent_missions.utils.gen_mission_plot import save_results, plot_results


def thermal_mission_problem(num_seg=5, transcription_order=3, meeting_altitude=20000., Q_env=0., Q_sink=0., Q_out=0., m_flow=0.1, opt_m_flow=False, opt_m_burn=False, opt_throttle=True, engine_heat_coeff=0., pump_heat_coeff=0., record=True):

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
    p.driver.options['dynamic_simul_derivs'] = True
    p.driver.options['dynamic_simul_derivs_repeats'] = 5

    phase = Phase('gauss-lobatto', ode_class=ThermalMissionODE,
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

    phase.set_state_options('m', fix_initial=True, lower=10.e3, upper=80e3,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    if opt_throttle:
        phase.add_control('throttle', val=1.0, lower=0.0, upper=1.0, dynamic=True, opt=True, rate_continuity=True)
    else:
        phase.add_control('throttle', val=1.0, dynamic=False, opt=False)
    phase.add_control('W0', val=10000., dynamic=False, opt=False, units='kg')

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
    if opt_m_flow:
        phase.add_control('m_flow', val=m_flow, lower=0.5, dynamic=True, opt=True, rate_continuity=True, ref=20.)
    else:
        phase.add_control('m_flow', val=m_flow, dynamic=True, opt=False)

    phase.add_control('Q_env', val=Q_env, dynamic=False, opt=False)
    phase.add_control('Q_sink', val=Q_sink, dynamic=False, opt=False)
    phase.add_control('Q_out', val=Q_out, dynamic=False, opt=False)

    # Constrain the temperature, 2nd derivative of fuel mass in the tank, and make
    # sure that the amount recirculated is at least 0, otherwise we'd burn
    # more fuel than we pumped.
    if opt_m_flow:
        phase.add_path_constraint('T', lower=0.)
        phase.add_path_constraint('T', upper=310., ref=300.)
        phase.add_path_constraint('T_o', lower=0., units='K')
        # phase.add_path_constraint('T_o', upper=302., units='K', ref=300.)
        # # phase.add_path_constraint('m_flow_rate', upper=0.)
        phase.add_path_constraint('m_recirculated', lower=0., upper=0., units='kg/s', ref=10.)
        phase.add_path_constraint('m_flow', lower=0., upper=40., ref=20., units='kg/s')

    p.setup(mode='fwd', check=True)
    if record:
        p.driver.add_recorder(SqliteRecorder('out.db'))

        p['phase.t_initial'] = 0.0
        p['phase.t_duration'] = 50.
        p['phase.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
        p['phase.states:h'] = phase.interpolate(ys=[100.0, meeting_altitude], nodes='disc')
        p['phase.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
        p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
        p['phase.states:m'] = phase.interpolate(ys=[20.e3, 16841.431], nodes='disc')
        # p['phase.controls:alpha'] = phase.interpolate(ys=[0.50, 0.50], nodes='all')

        # Give initial values for the phase states, controls, and time
        p['phase.states:T'] = 300.

    return p


if __name__ == '__main__':
    p = thermal_mission_problem(num_seg=12, transcription_order=3, m_flow=30., opt_m_flow=True, Q_env=0.e3, Q_sink=100.e3, Q_out=0.e3, engine_heat_coeff=60.e3)
    p.run_driver()

    save_results(p, 'test.pkl')
    plot_results('test.pkl', save_fig=True)
