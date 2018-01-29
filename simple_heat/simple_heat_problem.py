from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE
import numpy as np

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['Major iterations limit'] = 2000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['Verify level'] = -1

# phase = RadauPseudospectralPhase(ode_function=SimpleHeatODE(), num_segments=5, transcription_order=3, compressed=False)
phase = GaussLobattoPhase(ode_function=SimpleHeatODE(), num_segments=10, transcription_order=9, compressed=False)

phase.set_time_options(opt_initial=False, opt_duration=False)

phase.set_state_options('m', lower=1., upper=10.)
phase.set_state_options('T', fix_initial=True)
phase.set_state_options('energy', fix_initial=True)
phase.set_objective('energy', loc='final')
# phase.set_objective('time')

phase.add_control('m_flow', opt=True, lower=0., upper=5., dynamic=True)
phase.add_control('m_burn', opt=False, dynamic=False)
phase.add_path_constraint('T', upper=1.)
# phase.add_path_constraint('fuel_burner.m_recirculated', lower=0.)
# phase.add_path_constraint('m', lower=1.)

p.model.add_subsystem('phase', phase)
# p.model.jacobian = DenseJacobian()

p.setup(check=True, force_alloc_complex=True)

p['phase.states:m'] = phase.interpolate(ys=[10., 10.], nodes='disc')
p['phase.states:T'] = phase.interpolate(ys=[1., 1.], nodes='disc')
p['phase.states:energy'] = 0.
p['phase.controls:m_flow'] = 10. #phase.interpolate(ys=[0., 1.], nodes='all')
p['phase.controls:m_burn'] = 0.
p['phase.t_initial'] = 0.
p['phase.t_duration'] = 1.


if 0:
    p.run_model()
    out = p.model.phase.simulate()
    p.check_partials(compact_print=True, method='cs')

    # from openmdao.api import view_model
    # view_model(p)

    m = out.get_values('m')
    time = out.get_values('time')
    T = out.get_values('T')
    m_flow = out.get_values('m_flow')
    energy = out.get_values('energy')

else:
    # p.run_model()
    # out = p.model.phase.simulate()
    p.run_driver()

    # p.check_totals()

    m = p.model.phase.get_values('m', nodes='all')
    time = p.model.phase.get_values('time', nodes='all')
    T = p.model.phase.get_values('T', nodes='all')
    m_flow = p.model.phase.get_values('m_flow', nodes='all')
    energy = p.model.phase.get_values('energy', nodes='all')

if 1:

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].plot(time, m)
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel('mass')

    axarr[1].plot(time, T)
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel('temp')

    axarr[2].plot(time, m_flow)
    axarr[2].set_xlabel('time')
    axarr[2].set_ylabel('m_flow')

    axarr[3].plot(time, energy)
    axarr[3].set_xlabel('time')
    axarr[3].set_ylabel('energy')

    plt.show()
