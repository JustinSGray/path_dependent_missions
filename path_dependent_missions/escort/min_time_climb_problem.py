from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from min_time_climb_ode import MinTimeClimbODE


def min_time_climb_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto',
                           top_level_densejacobian=True, meeting_altitude=15000.):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = -1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.90
        p.driver.opt_settings['Major step limit'] = 0.5

    phase = Phase('gauss-lobatto', ode_class=MinTimeClimbODE,
                        num_segments=num_seg,
                        transcription_order=transcription_order,
                        compressed=False)

    p.model.add_subsystem('phase', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('Isp', val=5000.0, units='s', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, dynamic=False, opt=False)

    phase.add_boundary_constraint('h', loc='final', equals=meeting_altitude, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Minimize time at the end of the phase
    phase.set_objective('time', loc='final', ref=100.0)
    # phase.set_objective('m', loc='final', ref=-10000.0)

    if top_level_densejacobian:
        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

    p.setup(mode='fwd', check=True)

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 298.46902
    p['phase.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['phase.states:h'] = phase.interpolate(ys=[100.0, meeting_altitude], nodes='disc')
    p['phase.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    p['phase.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='all')

    return p


if __name__ == '__main__':
    p = min_time_climb_problem(optimizer='SNOPT', num_seg=10, transcription_order=3)
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

    # plt.tight_layout()
    # plt.savefig('min_time.pdf', bbox_inches='tight')
    plt.show()
