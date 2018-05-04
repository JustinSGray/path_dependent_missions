from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, view_model, CSCJacobian, DirectSolver

from dymos import Phase
from mod2_power_ode import X57Mod2PowerODE

class TestX57Mod2PowerODE(unittest.TestCase):

    def test_power_ode_climb(self):

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['Major iterations limit'] = 500
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-8
        p.driver.opt_settings['Verify level'] = 1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.options['dynamic_simul_derivs_repeats'] = 5

        phase = Phase(transcription='radau-ps', ode_class=X57Mod2PowerODE,
                                   num_segments=1,
                                   transcription_order=3)

        p.model.add_subsystem('phase', phase)

        phase.set_time_options(opt_initial=False, duration_bounds=(10., 10.), duration_scaler=1.0)

        phase.set_state_options('m', fix_initial=True, lower=0.0, units='kg')
        phase.set_state_options('h', fix_initial=True, lower=0.0, units='m')
        phase.set_state_options('r', fix_initial=True, lower=0.0, units='m')
        phase.set_state_options('TAS', fix_initial=True, lower=0.0, units='m/s')

        phase.add_control(name='h_dot', val=2., dynamic=True, opt=True, rate_param='h_dot_rate', units='m/s')
        phase.add_control(name='r_dot', val=2., dynamic=True, opt=True, rate_param='r_dot_rate', units='m/s')
        phase.add_control(name='S', val=49.2386, dynamic=False, opt=False, units='m**2')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', ref=100.0)
        # phase.add_objective('m', loc='final', ref=-10000.0)

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')
        T0 = 24 + 273.0

        p['phase.t_initial'] = 0.0
        p['phase.t_duration'] = 660

        p['phase.states:h'] = phase.interpolate(ys=[8000., 8000], nodes='disc')
        p['phase.states:r'] = phase.interpolate(ys=[0, 20], nodes='disc')
        p['phase.states:TAS'] = phase.interpolate(ys=[300., 300.], nodes='all')
        p['phase.states:m'] = 19000.0

        p.run_model()

        p.model.list_inputs(print_arrays=True)
        p.model.list_outputs(print_arrays=True, residuals=True)


        from path_dependent_missions.utils.gen_mission_plot import plot_results
        # plot_results(p, ['h', 'm', 'TAS'])

if __name__ == '__main__':

    unittest.main()
