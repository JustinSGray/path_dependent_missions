from __future__ import print_function, division, absolute_import
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian, SqliteRecorder

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.escort.min_time_climb_ode import MinTimeClimbODE

_phase_map = {'gauss_lobatto': GaussLobattoPhase,
              'radau_ps': RadauPseudospectralPhase}


def reg_time_climb_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss_lobatto', alpha_guess=False,
                           top_level_densejacobian=True, simul_derivs=False,
                           thrust_model='bryson'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    # p.driver.options['simul_derivs'] = simul_derivs
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-9
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-9
        p.driver.opt_settings['Verify level'] = -1
        # p.driver.opt_settings['Function precision'] = 1.0E-6
        # p.driver.opt_settings['Linesearch tolerance'] = 0.10
        # p.driver.opt_settings['Major step limit'] = 0.5


    phase_class = _phase_map[transcription]

    phase = phase_class(ode_function=MinTimeClimbODE(thrust_model=thrust_model),
                        num_segments=num_seg,
                        transcription_order=transcription_order,
                        compressed=False)

    p.model.add_subsystem('phase', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 1e5),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, fix_final=True, lower=0, upper=1.0E8,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    phase.set_state_options('h', fix_initial=True, lower=0, upper=14000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', fix_initial=True, lower=0.01,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    phase.set_state_options('m', fix_initial=True, lower=1e3, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, lower=0., upper=1., dynamic=True, opt=True)

    phase.add_boundary_constraint('h', loc='final', equals=0., scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('r', loc='final', equals=1e5, units=None)
    # phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='aero.mach', lower=0.01, upper=.9)
    phase.add_path_constraint(name='prop.m_dot', upper=0.)
    phase.add_path_constraint(name='flight_dynamics.r_dot', lower=0.)
    phase.add_path_constraint(name='m', lower=1e4)

    phase.set_objective('time', loc='final', ref=10.0)
    # phase.set_objective('m', loc='final', ref=-100.0)
    # phase.set_objective('r', loc='final', ref=-100000.0)

    if top_level_densejacobian:
        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

    # p.driver.add_recorder(SqliteRecorder('out.db'))
    p.setup(mode='fwd', check=True)
    # from openmdao.api import view_model
    # view_model(p)

    hxs = np.linspace(0., 100.)
    hys = np.ones(hxs.shape) * 1e4
    hys[0] = hys[-1] = 0.

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 100.
    # p['phase.states:r'] = phase.interpolate(ys=[0.0, 1.e6], nodes='disc')
    p['phase.states:h'] = phase.interpolate(xs=hxs, ys=hys, nodes='disc')
    p['phase.states:v'] = phase.interpolate(ys=[200., 200.], nodes='disc')
    p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase.states:m'] = phase.interpolate(ys=[2e4, 1e4], nodes='disc')
    p['phase.controls:alpha'] = phase.interpolate(ys=[0., 0.], nodes='all')

    return p


if __name__ == '__main__':
    p = reg_time_climb_problem(optimizer='SNOPT', num_seg=3, transcription_order=3, thrust_model='smt')

    p.run_model()
    exp_out = p.model.phase.simulate(times='disc')
    p['phase.states:r'] = np.atleast_2d(exp_out.get_values('r')).T
    p['phase.states:h'] = np.atleast_2d(exp_out.get_values('h')).T
    p['phase.states:v'] = np.atleast_2d(exp_out.get_values('v')).T
    p['phase.states:gam'] = np.atleast_2d(exp_out.get_values('gam')).T
    p['phase.states:m'] = np.atleast_2d(exp_out.get_values('m')).T
    # p.check_partials(compact_print=True)
    # exit()
    p.run_driver()

    import numpy as np
    phase = p.model.phase

    import matplotlib.pyplot as plt

    time = p.model.phase.get_values('time', nodes='all')
    m = p.model.phase.get_values('m', nodes='all')
    mach = p.model.phase.get_values('aero.mach', nodes='all')
    h = p.model.phase.get_values('h', nodes='all')
    r = p.model.phase.get_values('r', nodes='all')

    f, axarr = plt.subplots(3, sharex=True)

    axarr[0].plot(r, h, 'ko')
    axarr[0].set_xlabel('range')
    axarr[0].set_ylabel('altitude')

    axarr[1].plot(r, mach, 'ko')
    axarr[1].set_xlabel('range')
    axarr[1].set_ylabel('mach')

    axarr[2].plot(r, m, 'ko')
    axarr[2].set_xlabel('range')
    axarr[2].set_ylabel('mass')

    if 1:
        exp_out = phase.simulate(times=np.linspace(0, p['phase.t_duration'], 50))
        time2 = exp_out.get_values('time')
        m2 = exp_out.get_values('m')
        mach2 = exp_out.get_values('aero.mach')
        h2 = exp_out.get_values('h')
        r2 = exp_out.get_values('r')
        axarr[0].plot(r2, h2)
        axarr[1].plot(r2, mach2)
        axarr[2].plot(r2, m2)

    plt.show()
