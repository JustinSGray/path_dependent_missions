from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from thermal_mission_ode import ThermalMissionODE


def thermal_mission_problem(num_seg=5, transcription_order=3, meeting_altitude=20000., Q_env=0., Q_sink=0., Q_out=0., m_flow=0.1, opt_m_flow=False, opt_m_burn=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 500
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-8
    p.driver.opt_settings['Verify level'] = -1
    p.driver.opt_settings['Function precision'] = 1.0E-6
    p.driver.opt_settings['Linesearch tolerance'] = 0.10

    phase = Phase('gauss-lobatto', ode_class=ThermalMissionODE,
                        num_segments=num_seg,
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

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, dynamic=False, opt=False)
    # phase.add_control('throttle', val=1.0, lower=0., upper=1., dynamic=True, opt=True, rate_continuity=True)
    phase.add_control('W0', val=10000., dynamic=False, opt=False, units='kg')

    phase.add_boundary_constraint('h', loc='final', equals=meeting_altitude, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1., units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    # phase.add_path_constraint(name='prop.m_dot', lower=-10.)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=100.0)
    # phase.add_objective('m', loc='final', ref=-10000.0)

    # Set the state options for mass, temperature, and energy.
    phase.set_state_options('T', fix_initial=True)
    phase.set_state_options('energy', fix_initial=True)

    # Allow the optimizer to vary the fuel flow
    if opt_m_flow:
        phase.add_control('m_flow', val=m_flow, lower=0.01, dynamic=True, opt=True, rate_continuity=True)
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
        phase.add_path_constraint('T', upper=333.)
        phase.add_path_constraint('T_o', lower=0., units='K')
        phase.add_path_constraint('T_o', upper=333., units='K')
        # phase.add_path_constraint('m_flow_rate', upper=0.)
        phase.add_path_constraint('m_recirculated', lower=0., units='kg/s')
        phase.add_path_constraint('m_flow', lower=0., upper=50.)

    p.setup(mode='fwd', check=True)

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 200.
    p['phase.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['phase.states:h'] = phase.interpolate(ys=[100.0, meeting_altitude], nodes='disc')
    p['phase.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    # p['phase.controls:alpha'] = phase.interpolate(ys=[0.50, 0.50], nodes='all')

    # Give initial values for the phase states, controls, and time
    p['phase.states:T'] = 300.
    p['phase.states:energy'] = 0.

    return p


if __name__ == '__main__':
    p = thermal_mission_problem(num_seg=20, transcription_order=3)
    p.run_model()
    p.run_driver()



    phase = p.model.phase

    # Plotting below here
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.switch_backend('agg')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f, axarr = plt.subplots(4, sharex=True)

    time = phase.get_values('time', nodes='all')
    m = phase.get_values('m', nodes='all')
    mach = phase.get_values('aero.mach', nodes='all')
    h = phase.get_values('h', nodes='all')
    r = phase.get_values('r', nodes='all') / 1e3
    alpha = phase.get_values('alpha', nodes='all')
    gam = phase.get_values('gam', nodes='all')
    throttle = phase.get_values('throttle', nodes='all')

    pad = 90

    axarr[0].plot(r, h, 'o', color=colors[0])
    axarr[0].set_ylabel('altitude, m', rotation='horizontal', horizontalalignment='left', labelpad=pad)
    # axarr[0].set_yticks([0., 15000.])

    axarr[1].plot(r, mach, 'o', color=colors[0])
    axarr[1].set_ylabel('mach', rotation='horizontal', horizontalalignment='left', labelpad=pad)

    axarr[2].plot(r, m, 'o', color=colors[0])
    axarr[2].set_ylabel('mass, kg', rotation='horizontal', horizontalalignment='left', labelpad=pad)
    # axarr[2].set_yticks([17695.2, 19000.])

    axarr[3].plot(r, alpha, 'o', color=colors[0])
    axarr[3].set_ylabel('alpha, deg', rotation='horizontal', horizontalalignment='left', labelpad=pad)

    # axarr[4].plot(r, gam, 'o', color=colors[i])
    # axarr[4].set_ylabel('gamma')

    # axarr[5].plot(r, throttle, 'o', color=colors[i])
    # axarr[5].set_ylabel('throttle')

    axarr[-1].set_xlabel('range, km')

    exp_out = phase.simulate(times=np.linspace(0, p['phase.t_duration'], 50))
    time2 = exp_out.get_values('time')
    m2 = exp_out.get_values('m')
    mach2 = exp_out.get_values('aero.mach')
    h2 = exp_out.get_values('h')
    r2 = exp_out.get_values('r') / 1e3
    alpha2 = exp_out.get_values('alpha')
    gam2 = exp_out.get_values('gam')
    throttle2 = exp_out.get_values('throttle')
    axarr[0].plot(r2, h2, color=colors[0])
    axarr[1].plot(r2, mach2, color=colors[0])
    axarr[2].plot(r2, m2, color=colors[0])
    axarr[3].plot(r2, alpha2, color=colors[0])
    # axarr[4].plot(r2, gam2, color=colors[i])
    # axarr[5].plot(r2, throttle2, color=colors[i])

    n_points = time.shape[0]
    col_data = np.zeros((n_points, 8))
    col_data[:, 0] = time[:, 0]
    col_data[:, 1] = m[:, 0]
    col_data[:, 2] = mach[:, 0]
    col_data[:, 3] = h[:, 0]
    col_data[:, 4] = r[:, 0]
    col_data[:, 5] = alpha[:, 0]
    col_data[:, 6] = gam[:, 0]
    col_data[:, 7] = throttle[:, 0]
    # np.savetxt('col_data_3.dat', col_data)

    n_points = time2.shape[0]
    sim_data = np.zeros((n_points, 8))
    sim_data[:, 0] = time2[:, 0]
    sim_data[:, 1] = m2[:, 0]
    sim_data[:, 2] = mach2[:, 0]
    sim_data[:, 3] = h2[:, 0]
    sim_data[:, 4] = r2[:, 0]
    sim_data[:, 5] = alpha2[:, 0]
    sim_data[:, 6] = gam2[:, 0]
    sim_data[:, 7] = throttle2[:, 0]
    # np.savetxt('sim_data_3.dat', sim_data)

    # plt.tight_layout()
    # plt.savefig('min_time.pdf', bbox_inches='tight')
    plt.show()
