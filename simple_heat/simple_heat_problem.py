from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE


p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['Major iterations limit'] = 2000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['Verify level'] = -1

phase = RadauPseudospectralPhase(ode_function=SimpleHeatODE(), num_segments=2, transcription_order=3, compressed=False)

phase.set_time_options(opt_initial=False, opt_duration=False, duration=5.)

phase.set_state_options('m', lower=0., upper=10.)
phase.set_state_options('T', lower=1., upper=1.)
phase.set_objective('m', loc='final', scaler=-1.)

phase.add_control('m_flow', opt=True, val=1., lower=.0001, dynamic=True)
phase.add_boundary_constraint('T_0', loc='final', upper=1.)
phase.add_path_constraint

p.model.add_subsystem('phase', phase)

p.setup()

p['phase.states:m'] = phase.interpolate(ys=[10, 10], nodes='disc')
p['phase.states:T'] = phase.interpolate(ys=[5, 5], nodes='disc')


# p.run_model()
# p.model.phase.simulate()

p.run_driver()

m = p.model.phase.get_values('m', nodes='all')
time = p.model.phase.get_values('time', nodes='all')
T_0 = p.model.phase.get_values('T_0', nodes='all')

print(m)

import matplotlib.pyplot as plt
plt.plot(time, m)
plt.show()
