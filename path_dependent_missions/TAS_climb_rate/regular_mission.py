from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian, IndepVarComp

from dymos import Phase

from dymos.examples.aircraft.aircraft_ode import AircraftODE
from path_dependent_missions.utils.gen_mission_plot import save_results, plot_results


def ex_aircraft_mission(transcription='radau-ps', num_seg=10, transcription_order=3,
                        optimizer='SNOPT', compressed=True):

        p = Problem(model=Group())
        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = -1
            p.driver.opt_settings['Major step limit'] = .5

        else:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5

        phase = Phase(transcription,
                      ode_class=AircraftODE,
                      num_segments=num_seg,
                      transcription_order=transcription_order,
                      compressed=compressed)

        p.model.add_subsystem('phase', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1000, 10000), duration_ref=1e4)

        phase.set_state_options('range', units='km', fix_initial=True, scaler=1.0E-3, defect_scaler=1e-2)
        phase.set_state_options('mass', fix_initial=True, scaler=1.0E-5, defect_scaler=1e-5)

        phase.add_control('alt', units='km', dynamic=True, lower=0.0, upper=15.0,
                          rate_param='climb_rate', rate_continuity=True,
                          rate2_param='climb_rate2', rate2_continuity=True, ref=10.)

        phase.add_control('TAS', units='m/s', dynamic=True, lower=0.0, upper=260.0,
                          rate_param='TAS_rate', rate_continuity=True, rate2_continuity=True, ref=200.)

        phase.add_boundary_constraint('alt', loc='initial', equals=.5, units='km')
        phase.add_boundary_constraint('alt', loc='final', equals=.5, units='km')
        phase.add_boundary_constraint('TAS', loc='initial', equals=160., units='m/s')
        phase.add_boundary_constraint('TAS', loc='final', equals=80., units='m/s')
        phase.add_boundary_constraint('range', loc='final', equals=2000., units='km', ref=1000.)
        phase.add_boundary_constraint('mass', loc='final', lower=177.e3, units='kg', ref=100.e3)

        phase.add_path_constraint('propulsion.tau', lower=0., upper=1.)
        phase.add_path_constraint('alt_rate2', upper=.0008, units='m/s**2', ref=0.03)

        phase.add_objective('time')
        # phase.add_objective('mass', loc='final', ref=-100.e3)
        # phase.add_objective('objective')

        p.model.jacobian = CSRJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')

        p['phase.t_initial'] = 0.0
        p['phase.t_duration'] = 10000.0
        p['phase.states:range'] = phase.interpolate(ys=(0, 25000), nodes='state_disc')
        p['phase.states:mass'] = phase.interpolate(ys=(200.e3, 200.e3), nodes='state_disc')
        p['phase.controls:TAS'] = phase.interpolate(ys=(250., 250.), nodes='control_disc')
        p['phase.controls:alt'][:] = 10.
        p['phase.controls:alt'][0] = .1
        p['phase.controls:alt'][-1] = .1


        return p


if __name__ == '__main__':

    p = ex_aircraft_mission(num_seg=10)

    p.run_driver()

    list_to_plot = ['range', 'mass', 'alt', 'TAS', 'propulsion.tau', 'mach_comp.mach']

    save_results(p, 'no_newton.pkl', list_to_save=list_to_plot, run_sim=False)
    plot_results('no_newton.pkl', save_fig=False, list_to_plot=list_to_plot)
