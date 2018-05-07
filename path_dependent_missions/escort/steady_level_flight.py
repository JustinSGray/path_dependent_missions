from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from path_dependent_missions.escort.min_time_climb_ode import MinTimeClimbODE
from path_dependent_missions.utils.gen_mission_plot import plot_results


def min_time_climb_problem(num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto',
                           top_level_densejacobian=True):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 500
    p.driver.opt_settings['Iterations limit'] = 1000000000
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-10
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-10
    p.driver.opt_settings['Verify level'] = 1
    p.driver.opt_settings['Function precision'] = 1.0E-6
    p.driver.opt_settings['Linesearch tolerance'] = .1
    p.driver.options['dynamic_simul_derivs'] = True
    p.driver.options['dynamic_simul_derivs_repeats'] = 5
    # p.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']


    phase = Phase(transcription, ode_class=MinTimeClimbODE,
                        num_segments=num_seg,
                        transcription_order=transcription_order)

    p.model.add_subsystem('phase', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(100, 100),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E8,
                            scaler=1.0E-4, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('h', lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', lower=10.0, upper=500.,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    phase.set_state_options('gam', lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='kg')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, dynamic=True, opt=True, lower=0., upper=1., rate_continuity=False)

    phase.add_path_constraint(name='h', lower=1e4, upper=1e4, ref=20000, units='m')
    phase.add_path_constraint(name='aero.mach', lower=1.2, upper=1.2)


    phase.add_objective('time', loc='final', ref=-100.0)

    p.model.jacobian = CSCJacobian()
    p.model.linear_solver = DirectSolver()

    p.setup(mode='fwd', check=True)

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 2000.
    p['phase.states:r'] = phase.interpolate(ys=[0.0, 1e6], nodes='disc')
    p['phase.states:h'][:] = 1e4
    p['phase.states:v'][:] = 250.
    p['phase.states:gam'][:] = 0.
    p['phase.states:m'] = phase.interpolate(ys=[50e3, 49e3], nodes='disc')

    return p


if __name__ == '__main__':
    p = min_time_climb_problem(transcription='radau-ps', num_seg=12, transcription_order=3)
    p.run_model()
    p.run_driver()

    plot_results(p, ['h', 'm', 'r', 'alpha', 'gam', 'throttle', 'aero.mach', 'throttle_rate', 'throttle_rate2'])
