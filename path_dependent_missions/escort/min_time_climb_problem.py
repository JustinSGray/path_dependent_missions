from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from min_time_climb_ode import MinTimeClimbODE

_phase_map = {'gauss-lobatto': GaussLobattoPhase,
              'radau-ps': RadauPseudospectralPhase}


def min_time_climb_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto',
                           top_level_densejacobian=True):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = -1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5

    phase_class = _phase_map[transcription]

    phase = phase_class(ode_function=MinTimeClimbODE(),
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
    phase.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, dynamic=False, opt=False)

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Minimize time at the end of the phase
    phase.set_objective('time', loc='final', ref=100.0)

    if top_level_densejacobian:
        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

    p.setup(mode='fwd', check=True)

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 298.46902
    p['phase.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['phase.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='disc')
    p['phase.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    p['phase.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='all')

    return p


if __name__ == '__main__':
    p = min_time_climb_problem(optimizer='SNOPT', num_seg=10, transcription_order=3)
    p.run_model()
    p.run_driver()




    # Plotting below here
    import numpy as np
    phase = p.model.phase
    import matplotlib.pyplot as plt

    time = p.model.phase.get_values('time', nodes='all')
    m = p.model.phase.get_values('m', nodes='all')
    mach = p.model.phase.get_values('aero.mach', nodes='all')
    h = p.model.phase.get_values('h', nodes='all')
    r = p.model.phase.get_values('r', nodes='all')
    alpha = p.model.phase.get_values('alpha', nodes='all')
    gam = p.model.phase.get_values('gam', nodes='all')
    throttle = p.model.phase.get_values('throttle', nodes='all')

    f, axarr = plt.subplots(6, sharex=True)

    axarr[0].plot(r, h, 'ko')
    axarr[0].set_ylabel('altitude')

    axarr[1].plot(r, mach, 'ko')
    axarr[1].set_ylabel('mach')

    axarr[2].plot(r, m, 'ko')
    axarr[2].set_ylabel('mass')

    axarr[3].plot(r, alpha, 'ko')
    axarr[3].set_ylabel('alpha')

    axarr[4].plot(r, gam, 'ko')
    axarr[4].set_ylabel('gamma')

    axarr[5].plot(r, throttle, 'ko')
    axarr[5].set_ylabel('throttle')
    axarr[5].set_xlabel('range')

    if 1:
        exp_out = phase.simulate(times=np.linspace(0, p['phase.t_duration'], 20))
        time2 = exp_out.get_values('time')
        m2 = exp_out.get_values('m')
        mach2 = exp_out.get_values('aero.mach')
        h2 = exp_out.get_values('h')
        r2 = exp_out.get_values('r')
        alpha2 = exp_out.get_values('alpha')
        gam2 = exp_out.get_values('gam')
        throttle2 = exp_out.get_values('throttle')
        axarr[0].plot(r2, h2)
        axarr[1].plot(r2, mach2)
        axarr[2].plot(r2, m2)
        axarr[3].plot(r2, alpha2)
        axarr[4].plot(r2, gam2)
        axarr[5].plot(r2, throttle2)

    plt.show()
