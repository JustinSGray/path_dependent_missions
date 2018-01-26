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

phase.set_time_options(opt_initial=False, opt_duration=False)

phase.set_state_options('m', lower=1., upper=10.)
phase.set_state_options('T', fix_initial=True)
phase.set_objective('m', loc='final', scaler=-1.)

phase.add_control('m_flow', opt=True, val=1., lower=.1, dynamic=False)
phase.add_control('m_burn', opt=True, val=.2, dynamic=True)
phase.add_path_constraint('T', upper=1.)
phase.add_path_constraint('fuel_burner.m_recirculated', lower=0.)
# phase.add_path_constraint('T_0', upper=1.)

p.model.add_subsystem('phase', phase)

p.setup(check=True)

p['phase.states:m'] = phase.interpolate(ys=[10, 1], nodes='disc')
p['phase.states:T'] = phase.interpolate(ys=[.5, 1.5], nodes='disc')

if 0:
    p.run_model()
    p.model.phase.simulate()
else:
    p.run_driver()


m = p.model.phase.get_values('m', nodes='all')
time = p.model.phase.get_values('time', nodes='all')
T = p.model.phase.get_values('T', nodes='all')
# T_0 = p.model.phase.get_values('T_0', nodes='all')

print(m)

if 1:

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(time, m)
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel('mass')

    axarr[1].plot(time, T)
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel('temp')

    # axarr[2].plot(time, T_0)
    # axarr[2].set_xlabel('time')
    # axarr[2].set_ylabel('out temp')

    plt.show()
