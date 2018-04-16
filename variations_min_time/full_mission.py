from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

import matplotlib.pyplot as plt

SHOW_PLOTS = True


def min_time_climb(optimizer='SLSQP', num_seg=3, transcription_order=5,
                   transcription='gauss-lobatto',
                   top_level_jacobian='csc'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Verify level'] = -1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        # p.driver.opt_settings['Major step limit'] = 0.5

    phase = Phase(transcription,
                  ode_class=MinTimeClimbODE,
                  num_segments=num_seg, transcription_order=transcription_order,
                  compressed=False)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 1e5),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-4, units='m')

    phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            defect_scaler=1e-1, scaler=1e1, units='rad')

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1e0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)
    # phase.add_control('throttle', val=1.0, dynamic=False, opt=False)
    phase.add_control('throttle', val=1., dynamic=True, opt=True, lower=0., upper=1., rate_continuity=True, scaler=1e0)

    phase.add_boundary_constraint('h', loc='final', equals=100, scaler=1.0E-3, units='m')
    # phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    # phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')
    phase.add_boundary_constraint('r', loc='final', equals=5e5, units='m')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    phase.add_path_constraint(name='alpha', lower=-8, upper=8)

    # phase.add_objective('time', loc='final', ref=100.0)
    phase.add_objective('m', loc='final', ref=-1000.0)

    if top_level_jacobian.lower() == 'csc':
        p.model.jacobian = CSCJacobian()
    elif top_level_jacobian.lower() == 'dense':
        p.model.jacobian = DenseJacobian()
    elif top_level_jacobian.lower() == 'csr':
        p.model.jacobian = CSRJacobian()

    p.model.linear_solver = DirectSolver()

    p.setup(mode='fwd', check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 1000.
    p['phase0.states:r'] = phase.interpolate(ys=[0.0, 5e5], nodes='disc')
    p['phase0.states:h'] = phase.interpolate(ys=[100.0, 100.0], nodes='disc')
    # p['phase0.states:h'][1:-1] = 1e4
    p['phase0.states:v'] = phase.interpolate(ys=[200., 200.], nodes='disc')
    p['phase0.states:gam'] = phase.interpolate(ys=[0.5, 0.4], nodes='disc')
    p['phase0.states:m'] = phase.interpolate(ys=[20e3, 18e3], nodes='disc')
    p['phase0.controls:alpha'] = phase.interpolate(ys=[1.5, 0.5], nodes='all')

    p.run_model()

    # exp_out = phase.simulate(times='disc')
    #
    # p['phase0.states:r'] = exp_out.get_values('r')
    # p['phase0.states:h'] = exp_out.get_values('h')
    # p['phase0.states:v'] = exp_out.get_values('v')
    # p['phase0.states:gam'] = exp_out.get_values('gam')
    # p['phase0.states:m'] = exp_out.get_values('m')
    # p['phase0.controls:alpha'] = phase.interpolate(ys=[0.50, 0.50], nodes='all')

    p.run_driver()

    if SHOW_PLOTS:
        phase = p.model.phase0

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        f, axarr = plt.subplots(7, sharex=True)

        time = phase.get_values('time', nodes='all')
        m = phase.get_values('m', nodes='all')
        mach = phase.get_values('aero.mach', nodes='all')
        h = phase.get_values('h', nodes='all')
        r = phase.get_values('r', nodes='all') / 1e3
        alpha = phase.get_values('alpha', nodes='all')
        gam = phase.get_values('gam', nodes='all')
        throttle = phase.get_values('throttle', nodes='all')

        pad = 90

        axarr[0].plot(time, h, 'o', color=colors[0])
        axarr[0].set_ylabel('altitude, m', rotation='horizontal', horizontalalignment='left', labelpad=pad)
        # axarr[0].set_yticks([0., 15000.])

        axarr[1].plot(time, mach, 'o', color=colors[0])
        axarr[1].set_ylabel('mach', rotation='horizontal', horizontalalignment='left', labelpad=pad)

        axarr[2].plot(time, m, 'o', color=colors[0])
        axarr[2].set_ylabel('mass, kg', rotation='horizontal', horizontalalignment='left', labelpad=pad)
        # axarr[2].set_yticks([17695.2, 19000.])

        axarr[3].plot(time, alpha, 'o', color=colors[0])
        axarr[3].set_ylabel('alpha, deg', rotation='horizontal', horizontalalignment='left', labelpad=pad)

        axarr[4].plot(time, gam, 'o', color=colors[0])
        axarr[4].set_ylabel('gamma', rotation='horizontal', horizontalalignment='left', labelpad=pad)

        axarr[5].plot(time, throttle, 'o', color=colors[0])
        axarr[5].set_ylabel('throttle', rotation='horizontal', horizontalalignment='left', labelpad=pad)

        axarr[6].plot(time, r, 'o', color=colors[0])
        axarr[6].set_ylabel('range', rotation='horizontal', horizontalalignment='left', labelpad=pad)

        axarr[-1].set_xlabel('time, s')

        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))
        time2 = exp_out.get_values('time')
        m2 = exp_out.get_values('m')
        mach2 = exp_out.get_values('aero.mach')
        h2 = exp_out.get_values('h')
        r2 = exp_out.get_values('r') / 1e3
        alpha2 = exp_out.get_values('alpha')
        gam2 = exp_out.get_values('gam')
        throttle2 = exp_out.get_values('throttle')
        axarr[0].plot(time2, h2, color=colors[0])
        axarr[1].plot(time2, mach2, color=colors[0])
        axarr[2].plot(time2, m2, color=colors[0])
        axarr[3].plot(time2, alpha2, color=colors[0])
        axarr[4].plot(time2, gam2, color=colors[0])
        axarr[5].plot(time2, throttle2, color=colors[0])
        axarr[6].plot(time2, r2, color=colors[0])

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

        plt.tight_layout()
        # plt.savefig('min_time.pdf', bbox_inches='tight')
        plt.show()


    return p


if __name__ == '__main__':
    p = min_time_climb(transcription='gauss-lobatto', optimizer='SNOPT',
                       num_seg=7, transcription_order=3, top_level_jacobian='csc')
