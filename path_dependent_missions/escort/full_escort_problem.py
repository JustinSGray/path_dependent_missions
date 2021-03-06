from __future__ import print_function, division, absolute_import
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, \
    SqliteRecorder

from dymos import Phase
from dymos.phases.components import PhaseLinkageComp

from min_time_climb_ode import MinTimeClimbODE
from path_dependent_missions.escort.read_db import read_db


def escort_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto', meeting_altitude=15000.):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 200
        p.driver.opt_settings['Iterations limit'] = 100000000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = -1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5
        p.driver.options['dynamic_simul_derivs'] = True

    climb = Phase('gauss-lobatto', ode_class=MinTimeClimbODE,
                        num_segments=num_seg,
                        transcription_order=transcription_order)

    climb.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    climb.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    climb.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    climb.set_state_options('v', fix_initial=True, lower=10.0, upper=500.,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    climb.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    climb.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    climb.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True)

    climb.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    # climb.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)
    climb.add_control('throttle', val=1.0, dynamic=False, opt=False)

    climb.add_boundary_constraint('h', loc='final', equals=meeting_altitude, scaler=1.0E-3, units='m')
    climb.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    climb.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    # climb.add_boundary_constraint('time', loc='final', equals=350.0, units='s')

    climb.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    climb.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    p.model.add_subsystem('climb', climb)




    escort = Phase('gauss-lobatto', ode_class=MinTimeClimbODE,
                        num_segments=num_seg*2,
                        transcription_order=transcription_order)

    escort.set_time_options(duration_bounds=(50, 1000), duration_ref=100.0)

    escort.set_state_options('r', lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    escort.set_state_options('h', lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    escort.set_state_options('v', lower=10.0, upper=500.,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    escort.set_state_options('gam', lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    escort.set_state_options('m', lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    escort.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True)

    escort.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    # escort.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)

    escort.add_control('throttle', val=1.0, lower=0., upper=1., opt=True, rate_continuity=True)
    # escort.add_control('throttle', val=1.0, dynamic=False, opt=False)

    escort.add_path_constraint(name='h', lower=meeting_altitude, upper=meeting_altitude, ref=meeting_altitude)
    escort.add_path_constraint(name='aero.mach', equals=1.0)

    # Maximize distance at the end of the escort
    escort.set_objective('r', loc='final', ref=-1e5)

    p.model.add_subsystem('escort', escort)



    descent = Phase('gauss-lobatto', ode_class=MinTimeClimbODE,
                        num_segments=2,
                        transcription_order=5)

    descent.set_time_options(duration_bounds=(10, 100), duration_ref=100.0)

    descent.set_state_options('r', lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    descent.set_state_options('h', lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    descent.set_state_options('v', lower=10.0, upper=500.,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    descent.set_state_options('gam', lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    descent.set_state_options('m', lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    descent.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True)

    descent.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    # descent.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)

    descent.add_control('throttle', val=1.0, lower=0., upper=1., opt=True)
    # descent.add_control('throttle', val=0., dynamic=False, opt=False)

    descent.add_boundary_constraint('m', loc='final', equal=14000.0, units='kg')
    descent.add_boundary_constraint('h', loc='final', equals=100.0, units='m')
    descent.add_boundary_constraint('aero.mach', loc='final', equals=.1, units='m')
    descent.add_path_constraint(name='aero.mach', upper=1.2)
    descent.add_path_constraint(name='alpha_rate', lower=-0.5, upper=0.5)

    # descent.set_objective('time', loc='final', ref=1e2)

    p.model.add_subsystem('descent', descent)




    # Connect the phases
    linkage_comp = PhaseLinkageComp()
    linkage_comp.add_linkage(name='L01', vars=['t'], units='s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['r'], units='m', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['h'], units='m', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['v'], units='m/s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['gam'], units='rad', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['m'], units='kg', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['alpha'], units='rad', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['throttle'], equals=0.0, linear=True)

    p.model.connect('climb.time++', 'linkages.L01_t:lhs')
    p.model.connect('escort.time--', 'linkages.L01_t:rhs')

    p.model.connect('climb.states:r++', 'linkages.L01_r:lhs')
    p.model.connect('escort.states:r--', 'linkages.L01_r:rhs')

    p.model.connect('climb.states:h++', 'linkages.L01_h:lhs')
    p.model.connect('escort.states:h--', 'linkages.L01_h:rhs')
    #
    p.model.connect('climb.states:v++', 'linkages.L01_v:lhs')
    p.model.connect('escort.states:v--', 'linkages.L01_v:rhs')
    #
    p.model.connect('climb.states:gam++', 'linkages.L01_gam:lhs')
    p.model.connect('escort.states:gam--', 'linkages.L01_gam:rhs')

    p.model.connect('climb.states:m++', 'linkages.L01_m:lhs')
    p.model.connect('escort.states:m--', 'linkages.L01_m:rhs')

    p.model.connect('climb.controls:alpha++', 'linkages.L01_alpha:lhs')
    p.model.connect('escort.controls:alpha--', 'linkages.L01_alpha:rhs')

    p.model.connect('climb.controls:throttle++', 'linkages.L01_throttle:lhs')
    p.model.connect('escort.controls:throttle--', 'linkages.L01_throttle:rhs')

    p.model.add_subsystem('linkages', linkage_comp)

    # Connect the phases
    linkage_comp = PhaseLinkageComp()
    linkage_comp.add_linkage(name='L01', vars=['t'], units='s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['r'], units='m', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['h'], units='m', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['v'], units='m/s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['gam'], units='rad', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['m'], units='kg', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['alpha'], units='rad', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['throttle'], equals=0.0, linear=True)

    p.model.connect('escort.time++', 'linkages_2.L01_t:lhs')
    p.model.connect('descent.time--', 'linkages_2.L01_t:rhs')

    p.model.connect('escort.states:r++', 'linkages_2.L01_r:lhs')
    p.model.connect('descent.states:r--', 'linkages_2.L01_r:rhs')

    p.model.connect('escort.states:h++', 'linkages_2.L01_h:lhs')
    p.model.connect('descent.states:h--', 'linkages_2.L01_h:rhs')
    #
    p.model.connect('escort.states:v++', 'linkages_2.L01_v:lhs')
    p.model.connect('descent.states:v--', 'linkages_2.L01_v:rhs')
    #
    p.model.connect('escort.states:gam++', 'linkages_2.L01_gam:lhs')
    p.model.connect('descent.states:gam--', 'linkages_2.L01_gam:rhs')

    p.model.connect('escort.states:m++', 'linkages_2.L01_m:lhs')
    p.model.connect('descent.states:m--', 'linkages_2.L01_m:rhs')

    p.model.connect('escort.controls:alpha++', 'linkages_2.L01_alpha:lhs')
    p.model.connect('descent.controls:alpha--', 'linkages_2.L01_alpha:rhs')

    p.model.connect('escort.controls:throttle++', 'linkages_2.L01_throttle:lhs')
    p.model.connect('descent.controls:throttle--', 'linkages_2.L01_throttle:rhs')

    p.model.add_subsystem('linkages_2', linkage_comp)




    p.model.linear_solver = DirectSolver(assemble_jac=True)
    p.model.options['assembled_jac_type'] = 'csc'

    # p.driver.add_recorder(SqliteRecorder('escort.db'))

    p.setup(mode='fwd', check=True)

    p['climb.t_initial'] = 0.0
    p['climb.t_duration'] = 298.46902
    p['climb.states:r'] = climb.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['climb.states:h'] = climb.interpolate(ys=[100.0, 20000.0], nodes='disc')
    p['climb.states:v'] = climb.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['climb.states:gam'] = climb.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['climb.states:m'] = climb.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    p['climb.controls:alpha'] = climb.interpolate(ys=[0.0, 0.0], nodes='all')

    p['escort.t_initial'] = 300.
    p['escort.t_duration'] = 1000.
    p['escort.states:r'] = escort.interpolate(ys=[111319.54, 400000.], nodes='disc')
    p['escort.states:h'] = escort.interpolate(ys=[20000., 20000.0], nodes='disc')
    p['escort.states:v'] = escort.interpolate(ys=[250., 250.], nodes='disc')
    p['escort.states:gam'] = escort.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['escort.states:m'] = escort.interpolate(ys=[16841.431, 15000.], nodes='disc')
    p['escort.controls:alpha'] = escort.interpolate(ys=[0.0, 0.0], nodes='all')

    p['descent.t_initial'] = 1000.
    p['descent.t_duration'] = 1100.
    p['descent.states:r'] = descent.interpolate(ys=[400000., 500000.], nodes='disc')
    p['descent.states:h'] = descent.interpolate(ys=[20000., 100.], nodes='disc')
    p['descent.states:v'] = descent.interpolate(ys=[283.159, 135.], nodes='disc')
    p['descent.states:gam'] = descent.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['descent.states:m'] = descent.interpolate(ys=[15000., 14500.], nodes='disc')
    p['descent.controls:alpha'] = descent.interpolate(ys=[0.0, 0.0], nodes='all')

    return p


if __name__ == '__main__':
    p = escort_problem(optimizer='SNOPT', num_seg=10, transcription_order=3)

    # p.run_model()

    p.run_driver()




    # Plotting below here
    import numpy as np
    import matplotlib.pyplot as plt

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f, axarr = plt.subplots(8, sharex=True)

    for i, phase in enumerate([p.model.climb, p.model.escort, p.model.descent]):

        time = phase.get_values('time', nodes='all')
        m = phase.get_values('m', nodes='all')
        mach = phase.get_values('aero.mach', nodes='all')
        h = phase.get_values('h', nodes='all')
        r = phase.get_values('r', nodes='all')
        alpha = phase.get_values('alpha', nodes='all')
        alpha_rate = phase.get_values('alpha_rate', nodes='all')
        alpha_rate2 = phase.get_values('alpha_rate2', nodes='all')
        gam = phase.get_values('gam', nodes='all')
        throttle = phase.get_values('throttle', nodes='all')

        axarr[0].plot(time, h, 'o', color=colors[i])
        axarr[0].set_ylabel('altitude')

        axarr[1].plot(time, mach, 'o', color=colors[i])
        axarr[1].set_ylabel('mach')

        axarr[2].plot(time, m, 'o', color=colors[i])
        axarr[2].set_ylabel('mass')

        axarr[3].plot(time, gam, 'o', color=colors[i])
        axarr[3].set_ylabel('gamma')

        axarr[4].plot(time, throttle, 'o', color=colors[i])
        axarr[4].set_ylabel('throttle')

        axarr[5].plot(time, alpha, 'o', color=colors[i])
        axarr[5].set_ylabel('alpha')

        axarr[6].plot(time, alpha_rate, 'o', color=colors[i])
        axarr[6].set_ylabel('alpha_rate')

        axarr[7].plot(time, alpha_rate2, 'o', color=colors[i])
        axarr[7].set_ylabel('alpha_rate2')

        axarr[-1].set_xlabel('time, sec')

        exp_out = phase.simulate(times=np.linspace(0, p['climb.t_duration'], 100))
        time2 = exp_out.get_values('time')
        m2 = exp_out.get_values('m')
        mach2 = exp_out.get_values('aero.mach')
        h2 = exp_out.get_values('h')
        r2 = exp_out.get_values('r')
        alpha2 = exp_out.get_values('alpha')
        alpha_rate = exp_out.get_values('alpha_rate')
        alpha_rate2 = exp_out.get_values('alpha_rate2')
        gam2 = exp_out.get_values('gam')
        throttle2 = exp_out.get_values('throttle')
        axarr[0].plot(time2, h2, color=colors[i])
        axarr[1].plot(time2, mach2, color=colors[i])
        axarr[2].plot(time2, m2, color=colors[i])
        axarr[3].plot(time2, gam2, color=colors[i])
        axarr[4].plot(time2, throttle2, color=colors[i])
        axarr[5].plot(time2, alpha2, color=colors[i])
        axarr[6].plot(time2, alpha_rate, color=colors[i])
        axarr[7].plot(time2, alpha_rate2, color=colors[i])

    plt.show()
